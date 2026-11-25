"""Step 6: who is who.  Combines chest colour (dark / light / skin), court side and previous
position.  Using position alone swaps names when partners cross, so appearance is the
primary cue within a side, with position continuity as the tie-breaker.

Slots (0-based): 0,1 = near team, 2,3 = far team.  Each slot has an appearance spec:
  shirt: "dark" | "light"      (near pair differs by shirt)
  shoes: "dark" | "white"      (far pair: both dark shirts, differ by shoes)
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from .court import to_court, CW, CL, NET_Y


def box_features(frame, b):
    x1, y1, x2, y2 = [int(v) for v in b[:4]]
    h, w = y2 - y1, x2 - x1
    H, W = frame.shape[:2]
    x1c, x2c = max(0, x1), min(W, x2)
    # chest/torso: 20-45 % of the height, central 50 %
    cy1, cy2 = y1 + int(0.20 * h), y1 + int(0.45 * h)
    cx1, cx2 = x1 + int(0.25 * w), x1 + int(0.75 * w)
    chest = frame[max(0, cy1):min(H, cy2), max(0, cx1):min(W, cx2)]
    f = {"L": np.nan, "skin": 0.0, "shoe_white": np.nan}
    if chest.size > 30:
        lab = cv2.cvtColor(chest, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        skin = (hsv[:, 0] < 25) & (hsv[:, 1] > 50) & (hsv[:, 1] < 180) & (hsv[:, 2] > 90) & (lab[:, 1] > 135)
        f["skin"] = float(skin.mean())
        nonskin = lab[~skin] if (~skin).sum() > 10 else lab
        # share of bright (light-shirt) pixels, 0-100; robust to printed numbers/logos
        f["L"] = float(100.0 * np.mean(nonskin[:, 0] > 160))
    # shoes: bottom 9 % of the box
    sy1 = y2 - max(3, int(0.09 * h))
    shoe = frame[max(0, sy1):min(H, y2 + 2), x1c:x2c]
    if shoe.size > 12:
        hsv = cv2.cvtColor(shoe, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        white = (hsv[:, 2] > 165) & (hsv[:, 1] < 60)
        f["shoe_white"] = float(white.mean())
    return f


def candidates(frame, boxes, H):
    """Person boxes whose feet stand on/near our court."""
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        if y2 - y1 < 60 or b[4] < 0.2:
            continue
        foot = np.array([[(x1 + x2) / 2, y2]])
        c = to_court(H, foot)[0]
        if not (-1.2 < c[0] < CW + 1.2 and -2.5 < c[1] < CL + 1.0):
            continue
        f = box_features(frame, b)
        f.update({"box": b[:4].copy(), "court": c, "conf": float(b[4]), "tid": int(b[5]),
                  "side": 0 if c[1] < NET_Y else 1})
        out.append(f)
    return out


def assign_all(video, people, Hs, slots, fps, log=print):
    """Returns per-frame list of 4 entries (dict or None) and a per-slot summary."""
    from .camera import frames
    cand_all = []
    for t, f in enumerate(frames(video)):
        cand_all.append(candidates(f, people[t], Hs[t]))
        if t % 200 == 0:
            log(f"  identity: features frame {t}")
    # per-track-id appearance (averaged over its lifetime -> robust to single-frame noise)
    agg = {}
    for cs in cand_all:
        for c in cs:
            a = agg.setdefault(c["tid"], {"L": [], "sw": [], "side": []})
            if c["skin"] < 0.35 and not np.isnan(c["L"]):
                a["L"].append(c["L"])
            if not np.isnan(c["shoe_white"]):
                a["sw"].append(c["shoe_white"])
            a["side"].append(c["side"])
    # thresholds from data: split near chests into dark/light, far shoes into dark/white
    nearL = [c["L"] for cs in cand_all for c in cs if c["side"] == 0 and c["skin"] < 0.35 and not np.isnan(c["L"])]
    farS = [c["shoe_white"] for cs in cand_all for c in cs if c["side"] == 1 and not np.isnan(c["shoe_white"])]
    # threshold between per-track medians (single frames are noisy: numbers, arms, blur)
    tmed = [np.median(a["L"]) for a in agg.values() if len(a["L"]) >= 10 and np.mean(a["side"]) < 0.5]
    Lthr = _two_means(tmed, 6.0) if len(tmed) >= 2 else _two_means(nearL, 6.0)
    smed = [np.median(a["sw"]) for a in agg.values() if len(a["sw"]) >= 10 and np.mean(a["side"]) > 0.5]
    Sthr = _two_means(smed, 0.2) if len(smed) >= 2 else _two_means(farS, 0.2)
    log(f"  identity: chest-L split {Lthr:.0f}, far shoe-white split {Sthr:.2f}")

    def app_cost(c, slot):
        spec = slots[slot]
        tid = agg.get(c["tid"])
        cost = 0.0
        if "shirt" in spec and spec.get("side") == 0:
            Ls = [c["L"]] if (c["skin"] < 0.35 and not np.isnan(c["L"])) else []
            Lt = np.median(tid["L"]) if tid and tid["L"] else np.nan
            if Ls and not np.isnan(Lt):
                L = 0.25 * Ls[0] + 0.75 * Lt
            elif Ls:
                L = Ls[0]
            elif not np.isnan(Lt):
                L = Lt
            else:
                L = None
            if L is not None:
                z = (L - Lthr) / 4.0
                cost += np.clip(z if spec["shirt"] == "dark" else -z, -2, 2)
        if "shoes" in spec and spec.get("side") == 1:
            fv = c["shoe_white"]
            sw = tid["sw"] if tid else []
            tv = np.median(sw) if len(sw) else np.nan
            mixed = len(sw) >= 10 and (np.percentile(sw, 75) - np.percentile(sw, 25)) > 0.15
            wt = 0.2 if mixed else 0.7  # an ID that switched people: trust this frame
            if not np.isnan(fv) and not np.isnan(tv):
                sv = [(1 - wt) * fv + wt * tv]
            else:
                sv = [v for v in (fv, tv) if not np.isnan(v)]
            if sv:
                s = sv[0]
                z = (s - Sthr) / 0.12
                cost += np.clip(z if spec["shoes"] == "dark" else -z, -2, 2)
        return cost

    res = []
    last = [None] * 4  # (court pos, t)
    for t, cs in enumerate(cand_all):
        # one box per person: suppress duplicates
        cs = _nms(cs)
        n = len(cs)
        out = [None] * 4
        if n:
            C = np.full((n, 4), 50.0)
            for i, c in enumerate(cs):
                for s in range(4):
                    side_c = 0.0 if c["side"] == slots[s]["side"] else 6.0
                    pos_c = 0.0
                    if last[s] is not None:
                        dtm = (t - last[s][1]) / fps
                        dist = np.linalg.norm(c["court"] - last[s][0])
                        pos_c = max(0.0, dist - 6.0 * dtm - 0.3) / 0.6
                    C[i, s] = side_c + 1.5 * app_cost(c, s) + pos_c
            r, k = linear_sum_assignment(C)
            for i, s in zip(r, k):
                if C[i, s] < 8.0:
                    out[s] = cs[i]
                    last[s] = (cs[i]["court"], t)
        res.append(out)
    return res


def _two_means(v, default):
    v = np.asarray([x for x in v if not np.isnan(x)])
    if len(v) < 2:
        return default
    a, b = np.percentile(v, 25), np.percentile(v, 75)
    for _ in range(30):
        thr = (a + b) / 2
        lo, hi = v[v <= thr], v[v > thr]
        if len(lo) == 0 or len(hi) == 0:
            break
        a, b = lo.mean(), hi.mean()
    return (a + b) / 2


def _nms(cs, thr=0.6):
    cs = sorted(cs, key=lambda c: -c["conf"])
    keep = []
    for c in cs:
        ok = True
        for k in keep:
            if _iou(c["box"], k["box"]) > thr:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / (u + 1e-9)


def fill_tracks(assign, fps, max_hold=1.0):
    """Per-slot arrays of court position and box, interpolated over short gaps."""
    T = len(assign)
    pos = np.full((4, T, 2), np.nan)
    box = np.full((4, T, 4), np.nan)
    for t, row in enumerate(assign):
        for s, c in enumerate(row):
            if c is not None:
                pos[s, t] = c["court"]
                box[s, t] = c["box"]
    seen = ~np.isnan(pos[..., 0])
    for s in range(4):
        idx = np.where(seen[s])[0]
        if len(idx) == 0:
            continue
        for arr in (pos[s], box[s]):
            for d in range(arr.shape[1]):
                arr[:, d] = np.interp(np.arange(T), idx, arr[idx, d])
        # mark long gaps as unreliable (held)
    held = np.zeros((4, T), bool)
    for s in range(4):
        idx = np.where(seen[s])[0]
        if len(idx) == 0:
            held[s] = True
            continue
        nearest = np.min(np.abs(np.arange(T)[:, None] - idx[None, :]), axis=1) if len(idx) < 4000 else 0
        held[s] = nearest > max_hold * fps
    # smooth positions (court metres) a little
    from scipy.ndimage import uniform_filter1d
    pos_s = uniform_filter1d(pos, 5, axis=1, mode="nearest")
    box_s = uniform_filter1d(box, 3, axis=1, mode="nearest")
    return pos_s, box_s, held, seen
