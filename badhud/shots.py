"""Step 8: shots.  A shot = the shuttle reverses or is kicked sideways at a player's racket, or an
audio hit (2-8 kHz onset) while the shuttle is not visible.  Shots always alternate teams; the
hitter is the team member nearer the shuttle.

Speed off the racket (drag-only flight, terminal velocity 6.8 m/s):
    v0 = (e^(kD) - 1) / (kT),  k = g / 6.8^2
D = distance hitter -> next hitter (m, on the court), T = flight time (s)."""
import numpy as np
from .ball import near_racket, flight_team
from .court import to_court

K = 9.81 / 6.8 ** 2
MIN_GAP, MAX_GAP = 0.33, 2.4
RALLY_BREAK = 3.2


def speed_kmh(D, T):
    if T <= 0.05 or D <= 0:
        return 0.0
    return (np.exp(K * D) - 1) / (K * T) * 3.6


def _ball_img(bpts, t):
    return bpts.get(int(t))


def detect(path, all_tr, bpts, C, Hs, pos, boxes, onsets, fps, T, cfg, log=print, cands_ball=None):
    slots = cfg["slots"]
    team = [s["side"] for s in slots]
    names = [s["name"] for s in slots]

    def img(t, p):
        q = C[int(t)] @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]

    def member_near(f, pt, want_team):
        best = None
        for s in range(4):
            if team[s] != want_team:
                continue
            b = boxes[s, int(np.clip(f, 0, T - 1))]
            if np.isnan(b).any():
                continue
            cx, cy = (b[0] + b[2]) / 2, b[1] + 0.3 * (b[3] - b[1])
            d = np.hypot(pt[0] - cx, pt[1] - cy) / max(40.0, b[3] - b[1])
            if best is None or d < best[0]:
                best = (d, s)
        return None if best is None else best[1]

    def nearest_ball(f, rad):
        for d in range(rad + 1):
            for g in (f - d, f + d):
                if g in bpts:
                    return bpts[g]
        return None

    # ---- 1. shuttle-path events: reversal / sideways kick at a racket
    pev = []
    for a, b in zip(path[:-1], path[1:]):
        ta, tb = int(a["t"][-1]), int(b["t"][0])
        if tb - ta > 1.6 * fps:
            continue
        pa, pb = img(ta, a["p"][-1]), img(tb, b["p"][0])
        va = a["p"][-1] - a["p"][max(0, len(a["p"]) - 4)]
        vb = b["p"][min(len(b["p"]) - 1, 3)] - b["p"][0]
        cos = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        if cos > 0.6:
            continue
        za, zb = near_racket(pa, list(boxes[:, ta])), near_racket(pb, list(boxes[:, tb]))
        if za is None and zb is None:
            continue
        z, f = (zb, tb) if (zb is not None and (za is None or zb[1] < za[1])) else (za, ta)
        pev.append({"f": float(f), "slot": z[0], "lo": ta, "hi": tb, "tr": b})
    # start of every flight that begins at a racket = a hit just before it
    for tr in path:
        t0 = int(tr["t"][0])
        z = near_racket(img(t0, tr["p"][0]), list(boxes[:, t0]))
        if z is not None and not any(e["lo"] <= t0 <= e["hi"] for e in pev):
            pev.append({"f": float(t0), "slot": z[0], "lo": t0 - 4, "hi": t0, "tr": tr})
    # depth cue: the flight after the hit grows (far team hit it) or shrinks (near team)
    if cands_ball is not None:
        for e in pev:
            ft = flight_team(e["tr"], cands_ball)
            if ft is not None and team[e["slot"]] != ft:
                p0 = img(int(e["tr"]["t"][0]), e["tr"]["p"][0])
                s2 = member_near(e["f"], p0, ft)
                if s2 is not None:
                    e["slot"] = s2

    # ---- 2. rally windows from the visible flights (or from the edit file)
    rallies = _rallies_from_path(path, fps, T, cfg)

    # ---- 3. candidates: audio onsets + path events (merged when they coincide)
    on = [(t * fps, s) for t, s in onsets]
    strs = np.array([s for _, s in on]) if on else np.zeros(1)
    med = float(np.median(strs))
    cands = []
    for f, s in on:
        ev = float(np.clip((s - 2.5) / 5.0, 0, 2.5))
        cands.append({"f": f, "slot": None, "ev": ev, "src": "audio"})
    for e in pev:
        m = [c for c in cands if c["src"] == "audio" and e["lo"] - 4 <= c["f"] <= e["hi"] + 2]
        if m:
            c = max(m, key=lambda c: c["ev"])
            c.update({"slot": e["slot"], "ev": c["ev"] + 3.0, "src": "ball+audio"})
        else:
            cands.append({"f": e["f"], "slot": e["slot"], "ev": 2.0, "src": "ball"})
    # a hit cannot happen while the shuttle is seen in mid-flight away from all players
    ok = []
    for c in cands:
        f = int(round(c["f"]))
        if c["slot"] is None:
            mid = [g for g in (f - 1, f, f + 1) if g in bpts]
            if mid and all(near_racket(bpts[g], list(boxes[:, g])) is None for g in mid):
                continue
        ok.append(c)
    cands = sorted(ok, key=lambda c: c["f"])

    # ---- 4. manual edits
    rem = [float(x) for x in (cfg.get("shots_remove") or [])]
    # shots_add at the time of a detected shot replaces it (= change the hitter)
    rem += [float(a["t"]) for a in (cfg.get("shots_add") or [])]
    cands = [c for c in cands if not any(abs(c["f"] / fps - r) < 0.2 for r in rem)]
    forced = []
    for add in cfg.get("shots_add") or []:
        f = float(add["t"]) * fps
        pl = add.get("player")
        sl = names.index(pl) if pl in names else None
        ft = {"near": 0, "far": 1}.get(pl) if sl is None else team[sl]
        forced.append({"f": f, "slot": sl, "ev": 100.0, "src": "manual", "fteam": ft})
    cands = sorted(cands + forced, key=lambda c: c["f"])

    # ---- 5. per rally: best alternating sequence (DP)
    out = []
    for (ra, rb) in rallies:
        cs = [c for c in cands if ra <= c["f"] <= rb]
        out += _best_sequence(cs, team, fps)
    # hitters for audio-only shots: team member nearer the shuttle (or the net)
    for e in out:
        if e["slot"] is None:
            f = int(round(e["f"]))
            bp = nearest_ball(f, 8)
            s = member_near(f, bp, e["team"]) if bp is not None else None
            e["slot"] = s if s is not None else _closest_on_team(pos, team, e["team"], f, None)
        e["slot"] = int(e["slot"])
    # ---- 6. speeds
    shots = []
    out.sort(key=lambda e: e["f"])
    for i, e in enumerate(out):
        r = next((k for k, (a, b) in enumerate(rallies) if a - 0.5 * fps <= e["f"] <= b + 0.5 * fps), None)
        nxt = out[i + 1] if i + 1 < len(out) else None
        same = nxt is not None and r is not None and rallies[r][0] - 0.5 * fps <= nxt["f"] <= rallies[r][1] + 0.5 * fps
        ph = pos[e["slot"], int(e["f"])]
        est = False
        if same:
            pn = pos[nxt["slot"], int(nxt["f"])]
            D = float(np.linalg.norm(ph - pn))
            Tt = (nxt["f"] - e["f"]) / fps
            target = pn
        else:
            # last shot of the rally: only if the shuttle is seen landing (floor point known)
            est = True
            land = [tr for tr in path if e["f"] < tr["t"][-1] <= (rallies[r][1] + 0.5 * fps if r is not None else T)
                    and _lands(tr)]
            if land:
                tl = int(land[-1]["t"][-1])
                target = to_court(Hs[tl], bpts[tl][None])[0] if tl in bpts else ph
                D = float(np.linalg.norm(ph - target))
                Tt = (tl - e["f"]) / fps
            else:
                target, D, Tt = ph, 0.0, 0.0
        kmh = speed_kmh(D, Tt) if Tt > 0 else None
        shots.append({"i": i, "t": e["f"] / fps, "f": int(round(e["f"])), "slot": int(e["slot"]),
                      "name": names[e["slot"]], "team": team[e["slot"]], "src": e["src"],
                      "D": D, "T": Tt, "kmh": (float(kmh) if kmh is not None else None), "est": est, "rally": r,
                      "target": [float(target[0]), float(target[1])], "from": [float(ph[0]), float(ph[1])]})
    # ---- 7. warnings
    warns = []
    for k, s in enumerate(shots):
        if s["kmh"] is not None and (s["kmh"] > 250 or s["kmh"] < 25):
            warns.append({"t": s["t"], "shot": k + 1, "msg": f"ความเร็ว {s['kmh']:.0f} km/h ผิดปกติ (เกิน 250 หรือ ต่ำกว่า 25)"})
        if k + 1 < len(shots) and shots[k + 1]["rally"] == s["rally"]:
            g = shots[k + 1]["t"] - s["t"]
            if g < MIN_GAP:
                warns.append({"t": s["t"], "shot": k + 1, "msg": f"ช็อต {k + 1}->{k + 2} ห่างกันแค่ {g:.2f} วิ (< 0.33)"})
            if g > MAX_GAP:
                warns.append({"t": s["t"], "shot": k + 1, "msg": f"ช็อต {k + 1}->{k + 2} ห่างกัน {g:.2f} วิ (> 2.4) อาจตกหล่นช็อต"})
            if shots[k + 1]["team"] == s["team"]:
                warns.append({"t": s["t"], "shot": k + 1, "msg": f"ช็อต {k + 1}->{k + 2} ฝั่งเดียวกันตีติดกัน"})
    log(f"     {len(shots)} shots in {len(rallies)} rally(s); {len(warns)} warnings")
    return {"shots": shots, "rallies": rallies, "warnings": warns}


def _best_sequence(cs, team, fps, shot_cost=0.25):
    """Choose hits that alternate teams, >= 0.33 s apart, maximising evidence; long gaps
    (> 2.4 s) are penalised.  Manual shots are always kept."""
    n = len(cs)
    if n == 0:
        return []
    NEG = -1e9
    # state per candidate and its team (0/1)
    best = np.full((n, 2), NEG)
    prev = np.full((n, 2, 2), -1, int)
    for j, c in enumerate(cs):
        teams = [c["fteam"]] if c.get("fteam") is not None else [0, 1]
        for tj in teams:
            # the racket zone suggests a team, but in 2-D a far player standing behind a near
            # player shares the zone, so it is only a preference
            pref = 0.0 if c["slot"] is None else (1.0 if team[c["slot"]] == tj else -0.5)
            gain = c["ev"] - shot_cost + pref
            best[j, tj] = gain  # start of the rally
            for i in range(j):
                gap = (c["f"] - cs[i]["f"]) / fps
                manual = c["src"] == "manual" or cs[i]["src"] == "manual"
                if gap < (0.15 if manual else MIN_GAP):
                    continue
                # skipping a manual shot in between is not allowed
                if any(cs[k]["src"] == "manual" for k in range(i + 1, j)):
                    continue
                pen = 0.0 if gap <= 1.7 else 2.5 * (gap - 1.7)
                for ti in (0, 1):
                    if best[i, ti] <= NEG / 2:
                        continue
                    # teams must alternate; breaking it is only a last resort
                    v = best[i, ti] + gain - pen - (0.0 if ti != tj else 25.0)
                    if v > best[j, tj]:
                        best[j, tj] = v
                        prev[j, tj] = (i, ti)
            if any(cs[k]["src"] == "manual" for k in range(0, j)) and prev[j, tj][0] < 0:
                best[j, tj] = min(best[j, tj], NEG / 4)
    # end: last chosen must be after every manual shot
    last_manual = max([k for k, c in enumerate(cs) if c["src"] == "manual"], default=-1)
    j, tj = max(((j, t) for j in range(max(0, last_manual), n) for t in (0, 1)), key=lambda x: best[x])
    seq = []
    while j >= 0:
        c = dict(cs[j])
        c["team"] = tj
        if c["slot"] is not None and team[c["slot"]] != tj:
            c["slot"] = None
        seq.append(c)
        j, tj = prev[j, tj]
    return seq[::-1]


def _lands(tr):
    """Flight that ends falling steeply and slowing down (shuttle reaching the floor)."""
    p = tr["p"]
    if len(p) < 6:
        return False
    v = np.diff(p[-6:], axis=0)
    return bool((v[:, 1] > 0).all() and abs(v[:, 0]).mean() < 0.6 * v[:, 1].mean())


def _rallies_from_path(path, fps, T, cfg):
    if cfg.get("rallies"):
        return [(float(a) * fps, float(b) * fps) for a, b in cfg["rallies"]]
    fl = [(int(tr["t"][0]), int(tr["t"][-1]), len(tr["t"]), _lands(tr)) for tr in path]
    groups = []
    landed = False
    for a, b, n, ld in sorted(fl):
        if groups and a - groups[-1][1] <= 3.0 * fps and not (landed and a - groups[-1][1] > 1.2 * fps):
            groups[-1][1] = max(groups[-1][1], b)
            groups[-1][2] += n
        else:
            groups.append([a, b, n])
        landed = ld
    # a rally needs real flights, not a single short blip
    groups = [g for g in groups if g[2] >= 15]
    res = [(max(0.0, a - 1.6 * fps), min(float(T - 1), b + 0.3 * fps)) for a, b, _ in groups]
    if cfg.get("rally_start") is not None and res:
        st = cfg["rally_start"]
        st = st if isinstance(st, list) else [st]
        for k, v in enumerate(st[:len(res)]):
            if v is not None:
                res[k] = (float(v) * fps, res[k][1])
    re = cfg.get("rally_end")
    if re is not None and res:
        re = re if isinstance(re, list) else [None] * (len(res) - 1) + [re]
        for k, v in enumerate(re[:len(res)]):
            if v is not None:
                res[k] = (res[k][0], float(v) * fps)
    return res


def _audio_between(on, f0, f1, strong):
    c = [(s, f) for f, s in on if f0 <= f <= f1]
    if not c:
        return None
    s, f = max(c)
    return f if s >= strong * 0.5 else None


def _closest_on_team(pos, team, want, f, avoid):
    mem = [s for s in range(4) if team[s] == want]
    if len(mem) == 1:
        return mem[0]
    # the member nearer the net is more often the one in play; fall back to first
    from .court import NET_Y
    d = [abs(pos[s, min(f, pos.shape[1] - 1), 1] - NET_Y) if not np.isnan(pos[s, min(f, pos.shape[1] - 1), 1]) else 99 for s in mem]
    return mem[int(np.argmin(d))]


def _rallies(out, bpts, fps, T, cfg):
    if cfg.get("rallies"):
        return [(float(a) * fps, float(b) * fps) for a, b in cfg["rallies"]]
    groups = []
    for e in out:
        if groups and (e["f"] - groups[-1][-1]["f"]) / fps <= RALLY_BREAK:
            groups[-1].append(e)
        else:
            groups.append([e])
    groups = [g for g in groups if len(g) >= 2] or groups
    res = []
    seen = sorted(bpts)
    for g in groups:
        a = g[0]["f"] - 0.2 * fps
        last = g[-1]["f"]
        after = [t for t in seen if last < t <= last + 2.5 * fps]
        b = (max(after) if after else last + 1.0 * fps) + 0.2 * fps
        res.append((max(0.0, a), min(float(T - 1), b)))
    if cfg.get("rally_start") is not None and res:
        res[0] = (float(cfg["rally_start"]) * fps, res[0][1])
    re = cfg.get("rally_end")
    if re is not None and res:
        if isinstance(re, list):
            for k, v in enumerate(re[:len(res)]):
                res[k] = (res[k][0], float(v) * fps)
        else:
            res[-1] = (res[-1][0], float(re) * fps)
    return res
