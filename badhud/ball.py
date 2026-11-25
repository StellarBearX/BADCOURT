"""Step 7: shuttle.  Small bright blobs that are brighter than frames t+-3 and t+-10, where each
neighbour frame is compensated for camera motion twice (background homography and court-floor
homography) and the pixel must beat both.  Detections are linked into tracklets and a single
path for the whole rally is chosen that may change direction only at a player's racket."""
import os, pickle
import numpy as np
import cv2
from .camera import frames

OFFS = (-10, -3, 3, 10)


def detect_candidates(video, C, Hs, people, work, log=print, thr=22):
    out = os.path.join(work, "ball_cand.pkl")
    if os.path.exists(out):
        return pickle.load(open(out, "rb"))
    gs = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames(video)]
    T = len(gs)
    hh, ww = gs[0].shape
    k3 = np.ones((3, 3), np.uint8)
    res = []
    for t in range(T):
        g = gs[t]
        best = np.full(g.shape, 255, np.int16)
        nn = 0
        for o in OFFS:
            n = t + o
            if n < 0 or n >= T:
                continue
            nn += 1
            A = C[t] @ np.linalg.inv(C[n])
            B = Hs[t] @ np.linalg.inv(Hs[n])
            wa = cv2.warpPerspective(gs[n], A, (ww, hh), borderMode=cv2.BORDER_REPLICATE)
            wb = cv2.warpPerspective(gs[n], B, (ww, hh), borderMode=cv2.BORDER_REPLICATE)
            ref = cv2.dilate(np.maximum(wa, wb), k3)
            best = np.minimum(best, g.astype(np.int16) - ref.astype(np.int16))
        if nn < 3:
            res.append(np.zeros((0, 6), np.float32))
            continue
        m = ((best > thr) & (g > 105)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        cands = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < 3 or a > 700 or max(w, h) > 60:
                continue
            if max(w, h) / max(1, min(w, h)) > 6:
                continue
            sl = (slice(y, y + h), slice(x, x + w))
            peak = float(best[sl][lab[sl] == i].max())
            bright = float(g[sl][lab[sl] == i].mean())
            cands.append([cen[i][0], cen[i][1], a, peak, bright, max(w, h)])
        # moving bodies produce most false blobs: drop those inside a person box (shrunk)
        if len(cands) and t < len(people) and len(people[t]):
            b = people[t][:, :4]
            w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            x1, x2 = b[:, 0] + 0.12 * w, b[:, 2] - 0.12 * w
            y1, y2 = b[:, 1] - 0.03 * h, b[:, 3] + 0.03 * h
            keep = []
            for c in cands:
                inside = (c[0] > x1) & (c[0] < x2) & (c[1] > y1) & (c[1] < y2)
                if not inside.any():
                    keep.append(c)
            cands = keep
        # too many blobs = global flicker/blur frame: keep the strongest
        cands = sorted(cands, key=lambda c: -c[3])[:40]
        res.append(np.array(cands, np.float32).reshape(-1, 6))
        if t % 100 == 0:
            log(f"  ball: frame {t} ({len(cands)} candidates)")
    pickle.dump(res, open(out, "wb"))
    return res


# ------------------------------------------------------------------ tracklets
def stabilise(cands, C):
    stab = []
    for t, c in enumerate(cands):
        if len(c) == 0:
            stab.append(np.zeros((0, 2)))
            continue
        p = np.c_[c[:, :2], np.ones(len(c))] @ np.linalg.inv(C[t]).T
        stab.append(p[:, :2] / p[:, 2:3])
    return stab


def drop_static(stab, win=8, rad=7.0, max_hits=3):
    """Flicker that stays put (lights, reflections) is not a flying shuttle."""
    T = len(stab)
    keep = []
    for t in range(T):
        P = stab[t]
        if len(P) == 0:
            keep.append(np.zeros(0, bool))
            continue
        cnt = np.zeros(len(P), int)
        for o in range(-win, win + 1):
            if o == 0 or not (0 <= t + o < T) or len(stab[t + o]) == 0:
                continue
            d = np.linalg.norm(P[:, None] - stab[t + o][None], axis=2).min(1)
            cnt += d < rad
        keep.append(cnt <= max_hits)
    return keep


def _predict(pts, t):
    ts = np.array([x[0] for x in pts], float)
    ps = np.array([x[1] for x in pts])
    k = min(len(ts), 6)
    ts, ps = ts[-k:], ps[-k:]
    if k >= 4:
        A = np.c_[np.ones(k), ts - ts[-1], (ts - ts[-1]) ** 2]
        coef = np.linalg.lstsq(A, ps, rcond=None)[0]
        dt = t - ts[-1]
        return coef[0] + coef[1] * dt + coef[2] * dt * dt, np.linalg.norm(coef[1])
    v = (ps[-1] - ps[0]) / max(1e-6, ts[-1] - ts[0])
    return ps[-1] + v * (t - ts[-1]), np.linalg.norm(v)


def _extend(stab, alive, seed, direction, max_miss=3):
    pts = list(seed) if direction > 0 else list(seed[::-1])
    T = len(stab)
    t = pts[-1][0]
    miss = 0
    while miss <= max_miss:
        t += direction
        if not (0 <= t < T):
            break
        pred, sp = _predict(pts, t)
        P = stab[t]
        if len(P):
            d = np.linalg.norm(P - pred, axis=1)
            d[~alive[t]] = 1e9
            j = int(np.argmin(d))
            gate = 9 + 0.18 * sp * (1 + miss)
            if d[j] < gate:
                pts.append((t, P[j], j))
                miss = 0
                continue
        miss += 1
    return pts if direction > 0 else pts[::-1]


def build_tracklets(cands, C, min_len=6):
    """Seed-and-extend: triplets consistent with constant velocity, extended both ways
    with a local quadratic (ballistic) prediction.  Longest tracks claim detections first."""
    stab = stabilise(cands, C)
    alive = drop_static(stab)
    T = len(stab)
    seeds = []
    for t in range(T - 4):
        Pi = stab[t]
        for i in np.where(alive[t])[0]:
            for d1 in (1, 2):
                Pj = stab[t + d1]
                if len(Pj) == 0:
                    continue
                dj = np.linalg.norm(Pj - Pi[i], axis=1) / d1
                for j in np.where((dj > 1.5) & (dj < 170) & alive[t + d1])[0]:
                    v = (Pj[j] - Pi[i]) / d1
                    for d2 in (1, 2):
                        tk = t + d1 + d2
                        if tk >= T or len(stab[tk]) == 0:
                            continue
                        dk = np.linalg.norm(stab[tk] - (Pj[j] + v * d2), axis=1)
                        dk[~alive[tk]] = 1e9
                        k = int(np.argmin(dk))
                        if dk[k] < 6 + 0.12 * np.linalg.norm(v) * d2:
                            seeds.append([(t, Pi[i], i), (t + d1, Pj[j], j), (tk, stab[tk][k], k)])
    tracks = []
    for sd in seeds:
        fw = _extend(stab, alive, sd, +1)
        full = _extend(stab, alive, fw, -1)
        if len(full) >= min_len:
            tracks.append(full)
    tracks.sort(key=lambda tr: -len(tr))
    used = set()
    out = []
    for tr in tracks:
        tr2 = [x for x in tr if (x[0], x[2]) not in used]
        if len(tr2) < min_len or len(tr2) < 0.8 * len(tr):
            continue
        ts = np.array([x[0] for x in tr2])
        ps = np.array([x[1] for x in tr2])
        if np.linalg.norm(ps.max(0) - ps.min(0)) < 30:
            continue
        for x in tr2:
            used.add((x[0], x[2]))
        for piece in split_at_turns(ts, ps, [(x[0], x[2]) for x in tr2]):
            if len(piece["t"]) >= 4:
                out.append(piece)
    return out, stab


def split_at_turns(ts, ps, idx, max_turn=55):
    """Split a tracklet where the direction changes sharply."""
    cuts = [0]
    for i in range(2, len(ps) - 2):
        v1 = ps[i] - ps[i - 2]
        v2 = ps[i + 2] - ps[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 6 or n2 < 6:
            continue
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
        if ang > max_turn and i - cuts[-1] >= 2:
            cuts.append(i)
    cuts.append(len(ps))
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 3:
            pieces.append({"t": ts[a:b], "p": ps[a:b], "idx": idx[a:b]})
    return pieces


def tracklet_quality(tr, win=7):
    """Smoothness: median residual (px) of local quadratic fits over sliding windows."""
    t, p = tr["t"].astype(float), tr["p"]
    n = len(t)
    if n < 4:
        return 5.0
    res = []
    for i in range(0, max(1, n - win + 1)):
        tt, pp = t[i:i + win], p[i:i + win]
        if len(tt) < 4:
            continue
        A = np.c_[np.ones_like(tt), tt - tt[0], (tt - tt[0]) ** 2]
        r = 0
        for d in range(2):
            coef, *_ = np.linalg.lstsq(A, pp[:, d], rcond=None)
            r += np.mean((pp[:, d] - A @ coef) ** 2)
        res.append(np.sqrt(r))
    return float(np.median(res)) if res else 5.0


# ------------------------------------------------------------------ path selection
def racket_zones(boxes_t):
    """Approximate racket reach around a player box (image coords)."""
    zones = []
    for b in boxes_t:
        if b is None or np.isnan(b).any():
            zones.append(None)
            continue
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        zones.append((x1 - 0.9 * w - 20, y1 - 0.55 * h - 20, x2 + 0.9 * w + 20, y1 + 0.75 * h))
    return zones


def near_racket(pt_img, boxes_t):
    best = None
    for s, z in enumerate(racket_zones(boxes_t)):
        if z is None:
            continue
        if z[0] <= pt_img[0] <= z[2] and z[1] <= pt_img[1] <= z[3]:
            cx, cy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
            d = np.hypot(pt_img[0] - cx, pt_img[1] - cy)
            if best is None or d < best[1]:
                best = (s, d)
    return best


def select_path(tracklets, C, boxes, fps, log=print, node_cost=5.0, turn_penalty=8.0):
    """One shuttle path for the whole clip (max-weight chain of non-overlapping tracklets).
    Node weight = flight evidence (detections away from player bodies count fully, inside a
    racket zone 0.3) minus a fixed cost, so short jitter is never worth including.
    Changing direction between tracklets is free only at the racket of one of the 4 players
    (or for a straight continuation); anywhere else it costs `turn_penalty`."""
    def img(t, p):
        q = C[t] @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]

    trs = [tr for tr in tracklets if tracklet_quality(tr) < 6.0]
    trs.sort(key=lambda tr: tr["t"][0])
    n = len(trs)
    if n == 0:
        return [], []
    w = np.zeros(n)
    for k, tr in enumerate(trs):
        zin = sum(near_racket(img(int(t), p), list(boxes[:, int(t)])) is not None for t, p in zip(tr["t"], tr["p"]))
        q = tracklet_quality(tr)
        smooth = 1.0 if q < 2.5 else 0.8 if q < 4 else 0.5
        span = np.linalg.norm(tr["p"].max(0) - tr["p"].min(0))
        w[k] = ((len(tr["t"]) - zin) + 0.3 * zin) * smooth * (1.0 if span > 60 else 0.6) - node_cost
    best = w.copy()
    prev = -np.ones(n, int)
    for j in range(n):
        if w[j] <= 0:
            continue
        tj0 = trs[j]["t"][0]
        for i in range(j):
            if best[i] <= 0 or w[i] <= 0:
                continue
            if trs[i]["t"][-1] >= tj0:
                continue
            gap = tj0 - trs[i]["t"][-1]
            pen = 0.0 if gap > 3.0 * fps else (0.0 if _compatible(trs[i], trs[j], C, boxes, img, fps) else turn_penalty)
            if best[i] + w[j] - pen > best[j]:
                best[j] = best[i] + w[j] - pen
                prev[j] = i
    path = []
    j = int(np.argmax(best))
    while j >= 0:
        path.append(j)
        j = prev[j]
    path = path[::-1]
    log(f"  ball: {n} tracklets, path uses {len(path)} ({sum(len(trs[k]['t']) for k in path)} detections)")
    return [trs[k] for k in path], trs


def _dir(tr, end):
    p = tr["p"]
    k = min(4, len(p) - 1)
    v = (p[-1] - p[-1 - k]) if end else (p[k] - p[0])
    return v / (np.linalg.norm(v) + 1e-9), np.linalg.norm(v) / k


def _compatible(a, b, C, boxes, img, fps):
    ta, pa = a["t"][-1], a["p"][-1]
    tb, pb = b["t"][0], b["p"][0]
    if tb <= ta:
        return False
    da, sa = _dir(a, True)
    db, sb = _dir(b, False)
    jump = np.linalg.norm(pb - pa)
    dtf = tb - ta
    # straight continuation (same flight, missed frames)
    pred = pa + da * sa * dtf
    if np.linalg.norm(pred - pb) < 30 + 0.5 * sa * dtf and da @ db > 0.5:
        return True
    # direction change => must happen at a racket (either end point near a player)
    for t, p in ((ta, pa), (tb, pb)):
        bt = boxes[:, int(t)] if boxes is not None else None
        if bt is not None and near_racket(img(int(t), p), list(bt)) is not None:
            # the hit happened between ta and tb: jump must be feasible (<= 140 px / frame)
            return jump < 140 * dtf + 60
    return False


def path_points(path, C):
    """Per-frame image position of the shuttle along the chosen path (only observed frames)."""
    pts = {}
    for tr in path:
        for t, p in zip(tr["t"], tr["p"]):
            q = C[int(t)] @ np.array([p[0], p[1], 1.0])
            pts[int(t)] = q[:2] / q[2]
    return pts


def flight_team(tr, cands, ratio=1.6):
    """Which team hit this flight, from the shuttle's apparent size trend:
    growing = coming toward the camera = hit by the far team (1); shrinking = near team (0)."""
    if len(tr["idx"]) < 8:
        return None
    sz = np.array([np.sqrt(cands[t][j][2]) for t, j in tr["idx"]])
    k = max(3, len(sz) // 3)
    a, b = np.median(sz[:k]), np.median(sz[-k:])
    if b / max(a, 1e-6) > ratio:
        return 1
    if a / max(b, 1e-6) > ratio:
        return 0
    return None
