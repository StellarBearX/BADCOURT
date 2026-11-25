"""Steps 4-5: court calibration (first frame) and per-frame court tracking.

Court model in metres: x across (0 = left doubles sideline, 6.10 = right),
y along the court from the near back line (0) to the far back line (13.40).
H maps model (x, y, 1) -> image pixel.
"""
import itertools
import numpy as np
import cv2
from scipy.ndimage import maximum_filter1d, map_coordinates
from scipy.optimize import least_squares

CW, CL = 6.10, 13.40
NET_Y = 6.70
XS = [0.0, 0.46, 3.05, 5.64, 6.10]
YS = [0.0, 0.76, 4.72, 8.68, 12.64, 13.40]


def model_segments(near_only=False):
    segs = []
    ymax = NET_Y if near_only else CL
    for y in YS:
        if y <= ymax:
            segs.append(((0, y), (CW, y)))
    for x in (0.0, 0.46, 5.64, 6.10):
        segs.append(((x, 0), (x, ymax)))
    segs.append(((3.05, 0), (3.05, 4.72)))
    if not near_only:
        segs.append(((3.05, 8.68), (3.05, CL)))
    return segs


def sample_model(near_only=False, step=0.05):
    pts = []
    for (x1, y1), (x2, y2) in model_segments(near_only):
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], 1))
    return np.concatenate(pts)


SAMPLES_ALL = sample_model(False)
SAMPLES_NEAR = sample_model(True)


def project(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    z = p[:, 2:3]
    out = p[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    return out, z[:, 0]


def to_court(H, pts):
    Hi = np.linalg.inv(H)
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(np.asarray(pts).reshape(-1, 2)))] @ Hi.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- line mask
def floor_limit(hsv):
    """Per-column row where the green wall ends (floor starts below)."""
    Hh, Ww = hsv.shape[:2]
    green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 90)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    rows = (np.arange(Hh)[:, None] * green).max(0)
    rows[rows == 0] = int(Hh * 0.45)
    return maximum_filter1d(rows, 121) + 8


def ridge(g, ds=(6, 11)):
    """Thin-bright-line response: brighter than both neighbours at +-d along some direction."""
    g = g.astype(np.int16)
    out = np.zeros_like(g)
    for d in ds:
        for dx, dy in ((0, d), (d, 0), (d, d), (d, -d)):
            a = np.roll(g, (dy, dx), (0, 1))
            b = np.roll(g, (-dy, -dx), (0, 1))
            out = np.maximum(out, np.minimum(g - a, g - b))
    return out


def line_mask(frame, boxes=None, floor_rows=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    lim = floor_limit(hsv) if floor_rows is None else floor_rows
    floor = np.arange(g.shape[0])[:, None] > lim[None, :]
    m = (ridge(g) > 10) & (hsv[..., 1] < 120) & floor
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            h = y2 - y1
            m[max(0, y1):min(m.shape[0], y2 + int(0.06 * h)),
              max(0, x1 - 10):min(m.shape[1], x2 + 10)] = False
    m = m.astype(np.uint8) * 255
    m = drop_small(m)
    return m, floor


def drop_small(m, min_span=40):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    span = np.hypot(st[:, 2], st[:, 3])
    keep = (span >= min_span) & (((st[:, 4] / np.maximum(span, 1)) < 16) | (span > 400))
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def dist_map(mask):
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


# ---------------------------------------------------------------- scoring
def overlap_score(H, mask_dt, floor, samples=SAMPLES_ALL, sigma=4.0):
    """White-line overlap in pixels: projected model-line length lying on white pixels,
    minus a penalty for visible floor length that lands off the lines."""
    p, z = project(H, samples)
    hh, ww = mask_dt.shape
    step = np.r_[np.hypot(*np.diff(p, axis=0).T), 0]
    step = np.where(step > 40, 0, step)
    ok = (z > 0) & (p[:, 0] >= 0) & (p[:, 0] < ww - 1) & (p[:, 1] >= 0) & (p[:, 1] < hh - 1)
    if ok.sum() < 20:
        return -1e9, 0
    pi = p[ok].astype(int)
    vis = floor[pi[:, 1], pi[:, 0]]
    d = mask_dt[pi[:, 1], pi[:, 0]][vis]
    w = step[ok][vis]
    if len(d) < 20:
        return -1e9, 0
    good = np.exp(-d ** 2 / (2 * sigma ** 2))
    return float((good * w).sum() - 0.5 * (w * (d > 3 * sigma)).sum()), int(w.sum())


def refine(H, mask_dt, floor, samples=SAMPLES_ALL, iters=2, max_d=25.0, prior=None, prior_w=0.0):
    """Pull H toward the white lines: least squares on the distance map (point-to-line)."""
    hh, ww = mask_dt.shape
    H = H / H[2, 2]
    for _ in range(iters):
        p, z = project(H, samples)
        ok = (z > 0) & (p[:, 0] >= 1) & (p[:, 0] < ww - 2) & (p[:, 1] >= 1) & (p[:, 1] < hh - 2)
        if ok.sum() < 30:
            return H
        pi = p[ok].astype(int)
        ok2 = floor[pi[:, 1], pi[:, 0]] & (mask_dt[pi[:, 1], pi[:, 0]] < max_d)
        sel = samples[ok][ok2]
        if len(sel) < 30:
            return H
        h0 = H.flatten()[:8]

        def res(h):
            Hh = np.append(h, 1).reshape(3, 3)
            q, _ = project(Hh, sel)
            d = map_coordinates(mask_dt, [q[:, 1].clip(0, hh - 1), q[:, 0].clip(0, ww - 1)], order=1, mode="nearest")
            if prior is not None and prior_w > 0:
                anchor = np.array([[0, 0], [CW, 0], [0, NET_Y], [CW, NET_Y]])
                a1, _ = project(Hh, anchor)
                a0, _ = project(prior, anchor)
                return np.concatenate([d, prior_w * (a1 - a0).ravel()])
            return d

        scale = np.abs(h0) * 1e-3 + 1e-9
        r = least_squares(res, h0, loss="soft_l1", f_scale=3.0, x_scale=scale, max_nfev=60, diff_step=1e-4)
        H = np.append(r.x, 1).reshape(3, 3)
        max_d = max(8.0, max_d * 0.5)
    return H


# ---------------------------------------------------------------- Hough detection
def _line_from_seg(s):
    x1, y1, x2, y2 = map(float, s)
    a, b = y2 - y1, x1 - x2
    n = np.hypot(a, b)
    return np.array([a / n, b / n, -(a * x1 + b * y1) / n])


def _intersect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def hough_lines(mask):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, 60, minLineLength=70, maxLineGap=25)
    if segs is None:
        return np.zeros((0, 4))
    return segs.reshape(-1, 4).astype(float)


def find_vp(segs, rng):
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    cand = np.where((np.minimum(ang, 180 - ang) > 9))[0]
    if len(cand) < 2:
        return None, None
    lines = np.array([_line_from_seg(s) for s in segs])
    mids = np.c_[(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2]
    best, best_s = None, 0
    pairs = [(i, j) for i in cand for j in cand if i < j]
    if len(pairs) > 400:
        idx = rng.choice(len(pairs), 400, replace=False)
        pairs = [pairs[k] for k in idx]
    for i, j in pairs:
        vp = _intersect(lines[i], lines[j])
        if vp is None or not np.isfinite(vp).all():
            continue
        s, sup = _vp_support(vp, segs, mids, lens)
        if s > best_s:
            best, best_s = vp, s
    if best is None:
        return None, None
    _, sup = _vp_support(best, segs, mids, lens)
    return best, sup


def _vp_support(vp, segs, mids, lens, tol_deg=1.2):
    d = np.c_[segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]]
    v = vp[None] - mids
    cos = np.abs((d * v).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    sup = (angle < tol_deg) & (np.minimum(ang, 180 - ang) > 7)
    return lens[sup].sum(), sup


def cluster_lines(segs, keys, tol):
    """Greedy 1D clustering of segments by key; returns list of (key, total_len, line)."""
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(keys)
    groups = []
    for i in order:
        if groups and keys[i] - groups[-1]["k"][-1] < tol:
            groups[-1]["i"].append(i)
            groups[-1]["k"].append(keys[i])
        else:
            groups.append({"i": [i], "k": [keys[i]]})
    out = []
    for g in groups:
        idx = np.array(g["i"])
        pts = np.concatenate([segs[idx, :2], segs[idx, 2:]])
        w = np.repeat(lens[idx], 2)
        # weighted total-least-squares line fit
        c = (pts * w[:, None]).sum(0) / w.sum()
        cov = ((pts - c).T * w) @ (pts - c)
        ev, evec = np.linalg.eigh(cov)
        n = evec[:, 0]
        line = np.array([n[0], n[1], -n @ c])
        out.append((float(np.average(np.array(g["k"]), weights=lens[idx])), float(lens[idx].sum()), line))
    return out


def detect_court(frame, boxes=None, rng=None, prior=None, want_debug=False):
    """Hough -> VP of long lines -> bottom two horizontals = back line & long service line
    -> cross-ratio matching of long lines to x = 0, .46, 3.05, 5.64, 6.10 -> H -> refine.
    Returns (H, score, debug)."""
    rng = rng or np.random.default_rng(0)
    mask, floor = line_mask(frame, boxes)
    dt = dist_map(mask)
    segs = hough_lines(mask)
    dbg = {"segs": segs, "mask": mask}
    if len(segs) < 4:
        return None, -1e9, dbg
    vp, sup = find_vp(segs, rng)
    if vp is None:
        return None, -1e9, dbg
    dbg["vp"] = vp
    hh, ww = mask.shape
    # long lines: cluster by angle as seen from VP
    ls = segs[sup]
    mids = np.c_[(ls[:, 0] + ls[:, 2]) / 2, (ls[:, 1] + ls[:, 3]) / 2]
    th = np.degrees(np.arctan2(mids[:, 0] - vp[0], mids[:, 1] - vp[1]))
    longs = [c for c in cluster_lines(ls, th, 0.8) if c[1] > 60]
    # horizontal-ish lines (not through VP)
    hs = segs[~sup]
    ang = np.degrees(np.arctan2(hs[:, 3] - hs[:, 1], hs[:, 2] - hs[:, 0]))
    ang = (ang + 90) % 180 - 90
    hs = hs[np.abs(ang) < 20]
    if len(hs) == 0 or len(longs) < 2:
        return None, -1e9, dbg
    # key: y where the segment's line crosses image centre column
    keys = []
    for s in hs:
        l = _line_from_seg(s)
        keys.append(-(l[0] * ww / 2 + l[2]) / l[1] if abs(l[1]) > 1e-6 else 0)
    horiz = [c for c in cluster_lines(hs, np.array(keys), 14) if c[1] > 120]
    horiz.sort(key=lambda c: -c[0])  # bottom first
    dbg["longs"] = longs
    dbg["horiz"] = horiz
    if len(horiz) < 2:
        return None, -1e9, dbg

    # horizontal hypotheses: bottom two = (0, 0.76) is the primary; also allow shifted pairs
    hyps = [(0, 1, 0.0, 0.76), (0, 1, 0.76, 4.72), (1, 2, 0.0, 0.76)]
    longs.sort(key=lambda c: c[0])  # angle from VP: larger angle = more to the right? sort below
    best = (None, -1e9)
    for hi, hj, ya, yb in hyps:
        if hj >= len(horiz):
            continue
        la, lb = horiz[hi][2], horiz[hj][2]
        pa = [(_intersect(l[2], la), l) for l in longs]
        pb = [(_intersect(l[2], lb), l) for l in longs]
        cols = [(a, b) for (a, _), (b, _) in zip(pa, pb) if a is not None and b is not None
                and np.all(np.abs(a) < 2e4) and np.all(np.abs(b) < 2e4)]
        cols.sort(key=lambda ab: ab[0][0])  # left to right along the lower line
        n = len(cols)
        for k in range(min(n, 5), 1, -1):
            for det in itertools.combinations(range(n), k):
                ta = np.array([cols[i][0][0] for i in det])
                for mod in itertools.combinations(range(5), k):
                    X = np.array([XS[m] for m in mod])
                    if k >= 4 and not _cross_ratio_ok(ta, X):
                        continue
                    src = np.concatenate([np.c_[X, np.full(k, ya)], np.c_[X, np.full(k, yb)]])
                    dst = np.concatenate([[cols[i][0] for i in det], [cols[i][1] for i in det]])
                    Hc, _ = cv2.findHomography(src, dst, 0)
                    if Hc is None:
                        continue
                    if not _plausible(Hc, ww, hh):
                        continue
                    s, _ = overlap_score(Hc, dt, floor)
                    if s > best[1]:
                        best = (Hc, s)
    H = best[0]
    if H is None:
        return None, -1e9, dbg
    H = refine(H, dt, floor, SAMPLES_ALL, iters=3, max_d=30)
    s, _ = overlap_score(H, dt, floor)
    dbg["dt"], dbg["floor"] = dt, floor
    return H, s, dbg


def _cross_ratio_ok(t, X, tol=0.12):
    # compare cross ratios of consecutive 4-tuples (projective invariant)
    for i in range(len(t) - 3):
        a, b, c, d = t[i:i + 4]
        A, B, C, D = X[i:i + 4]
        cr1 = ((c - a) * (d - b)) / ((c - b) * (d - a) + 1e-9)
        cr2 = ((C - A) * (D - B)) / ((C - B) * (D - A) + 1e-9)
        if abs(cr1 - cr2) / abs(cr2) > tol:
            return False
    return True


def _plausible(H, ww, hh):
    """Near court corners must lie below the far ones and the camera must be behind the near line."""
    p, z = project(H, np.array([[0, 0], [CW, 0], [0, NET_Y], [CW, NET_Y], [0, CL], [CW, CL]]))
    if (z <= 0).any():
        return False
    if not (p[0, 1] > p[2, 1] > p[4, 1] and p[1, 1] > p[3, 1] > p[5, 1]):
        return False
    if p[1, 0] <= p[0, 0] or p[3, 0] <= p[2, 0]:
        return False
    # net width in pixels should be sensible
    w = p[3, 0] - p[2, 0]
    return 150 < w < 6 * ww


def draw_court(img, H, color=(0, 255, 255), thick=3, near_only=False):
    for (x1, y1), (x2, y2) in model_segments(near_only):
        pts = np.stack([np.linspace(x1, x2, 40), np.linspace(y1, y2, 40)], 1)
        p, z = project(H, pts)
        p = p[z > 0]
        if len(p) > 1:
            cv2.polylines(img, [p.astype(np.int32)], False, color, thick, cv2.LINE_AA)
    # net line on floor
    p, _ = project(H, np.array([[0, NET_Y], [CW, NET_Y]]))
    cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- per-frame tracking
NEAR_CORNERS = np.array([[0, 0], [CW, 0], [0, NET_Y], [CW, NET_Y]])


def corner_shift(Ha, Hb):
    a, _ = project(Ha, NEAR_CORNERS)
    b, _ = project(Hb, NEAR_CORNERS)
    return float(np.linalg.norm(a - b, axis=1).mean())


# ---------------------------------------------------------------- camera pose model
CX, CY = 540.0, 960.0


def focal_from_H(H):
    """Focal length (px) from the plane homography, principal point at the image centre."""
    T = np.array([[1, 0, -CX], [0, 1, -CY], [0, 0, 1]])
    Hc = T @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    ws = []
    d = h1[0] * h2[0] + h1[1] * h2[1]
    if abs(d) > 1e-12:
        ws.append(-h1[2] * h2[2] / d)
    d = (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)
    if abs(d) > 1e-12:
        ws.append((h2[2] ** 2 - h1[2] ** 2) / d)
    fs = [1 / np.sqrt(w) for w in ws if w > 0]
    return float(np.clip(np.median(fs), 600, 4000)) if fs else 1600.0


def Kmat(f):
    return np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.0]])


def pose_from_H(H, f):
    A = np.linalg.inv(Kmat(f)) @ H
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if A[2, 2] * lam < 0:  # court in front of the camera
        lam = -lam
    r1, r2, t = A[:, 0] * lam, A[:, 1] * lam, A[:, 2] * lam
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    rvec, _ = cv2.Rodrigues(R)
    return np.r_[rvec.ravel(), t]


def H_from_pose(p, f):
    R, _ = cv2.Rodrigues(p[:3])
    H = Kmat(f) @ np.c_[R[:, 0], R[:, 1], p[3:6]]
    return H / H[2, 2]


def cam_center(p):
    R, _ = cv2.Rodrigues(p[:3])
    return -R.T @ p[3:6]


def refine_pose(p0, f, mask_dt, floor, samples, iters=2, max_d=25.0, c_prior=None, c_w=0.0,
                r_prior=None, r_w=0.0):
    """Pull the camera pose (3 rotation + 3 translation) toward the white lines."""
    hh, ww = mask_dt.shape
    p = p0.copy()
    for _ in range(iters):
        H = H_from_pose(p, f)
        q, z = project(H, samples)
        ok = (z > 0) & (q[:, 0] >= 1) & (q[:, 0] < ww - 2) & (q[:, 1] >= 1) & (q[:, 1] < hh - 2)
        if ok.sum() < 30:
            return p
        qi = q[ok].astype(int)
        ok2 = floor[qi[:, 1], qi[:, 0]] & (mask_dt[qi[:, 1], qi[:, 0]] < max_d)
        sel = samples[ok][ok2]
        if len(sel) < 30:
            return p

        def res(x):
            Hh = H_from_pose(x, f)
            qq, _ = project(Hh, sel)
            d = map_coordinates(mask_dt, [qq[:, 1].clip(0, hh - 1), qq[:, 0].clip(0, ww - 1)], order=1, mode="nearest")
            extra = []
            if c_prior is not None and c_w > 0:
                extra.append(c_w * (cam_center(x) - c_prior))
            if r_prior is not None and r_w > 0:
                extra.append(r_w * (x[:3] - r_prior))
            return np.concatenate([d] + extra) if extra else d

        r = least_squares(res, p, loss="soft_l1", f_scale=3.0, max_nfev=50,
                          x_scale=np.r_[np.full(3, 0.01), np.full(3, 0.05)], diff_step=1e-5)
        p = r.x
        max_d = max(8.0, max_d * 0.5)
    return p


def plausible_pose(p):
    c = cam_center(p)
    # hand-held camera behind/near the near back line, at chest-to-head height
    return (0.6 < abs(c[2]) < 2.8) and (-7.0 < c[1] < 4.0) and (-4.0 < c[0] < CW + 4.0)


def rot_diff_deg(p, q):
    Ra, _ = cv2.Rodrigues(p[:3])
    Rb, _ = cv2.Rodrigues(q[:3])
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def track_court(video, people, Ms, work, H0=None, log=print):
    """Predict from camera motion, pull toward the white lines of the near half, and every
    2 frames re-read the court from scratch; keep whichever overlaps the white lines more.
    The court is parametrised as a physical camera pose (fixed focal length), so it cannot
    deform into impossible shapes, and the camera centre may only move slowly."""
    import os, pickle
    from .camera import frames
    out = os.path.join(work, "court.pkl")
    if os.path.exists(out):
        return pickle.load(open(out, "rb"))
    Hs, info, poses = [], [], []
    rng = np.random.default_rng(0)
    f = focal_from_H(H0)
    p = None
    for t, fr in enumerate(frames(video)):
        boxes = people[t] if t < len(people) else None
        mask, floor = line_mask(fr, boxes)
        dt = dist_map(mask)
        src = "track"
        if p is None:
            p = pose_from_H(H0, f)
            p = refine_pose(p, f, dt, floor, SAMPLES_ALL, iters=3, max_d=30)
            src = "detect"
        else:
            Hp = Ms[t] @ H_from_pose(p, f)
            pp = pose_from_H(Hp / Hp[2, 2], f)
            c_prev = cam_center(p)
            # camera centre: keep near the previous one (hand-held, slow drift)
            pp[3:6] = -cv2.Rodrigues(pp[:3])[0] @ c_prev
            pt = refine_pose(pp, f, dt, floor, SAMPLES_NEAR, iters=1, max_d=20,
                             c_prior=c_prev, c_w=40.0, r_prior=pp[:3], r_w=300.0)
            if not plausible_pose(pt):
                pt = pp
            st, _ = overlap_score(H_from_pose(pt, f), dt, floor)
            p = pt
            if t % 2 == 0:
                Hd, sd, _ = detect_court(fr, boxes, rng)
                if Hd is not None:
                    pd = pose_from_H(Hd, f)
                    pd = refine_pose(pd, f, dt, floor, SAMPLES_ALL, iters=2, max_d=20)
                    sd, _ = overlap_score(H_from_pose(pd, f), dt, floor)
                    ok = plausible_pose(pd) and np.linalg.norm(cam_center(pd) - c_prev) < 1.2 \
                        and rot_diff_deg(pd, pp) < 8
                    if ok and sd > st:
                        p, src = pd, "detect"
        H = H_from_pose(p, f)
        s, n = overlap_score(H, dt, floor)
        Hs.append(H)
        poses.append(p.copy())
        info.append({"src": src, "score": s, "n": n})
        if t % 50 == 0:
            c = cam_center(p)
            log(f"  court: frame {t} score {s:.0f} ({src}) cam=({c[0]:.1f},{c[1]:.1f},{abs(c[2]):.1f})m")
    res = {"H_raw": np.array(Hs), "info": info, "poses": np.array(poses), "f": f}
    pickle.dump(res, open(out, "wb"))
    return res


def smooth_court(Hs, C, win=7):
    """Remove jitter: express each H in frame-0 coordinates (G = C^-1 H), smooth the projected
    reference points over time, then map back with the camera motion."""
    from scipy.signal import savgol_filter
    ref = np.array([[0, 0], [CW, 0], [0, NET_Y], [CW, NET_Y], [0, 3.0], [CW, 3.0]])
    P = []
    for H, c in zip(Hs, C):
        G = np.linalg.inv(c) @ H
        p, _ = project(G, ref)
        P.append(p)
    P = np.array(P)
    n = len(P)
    w = min(win if win % 2 else win + 1, n - (1 - n % 2))
    if w >= 5:
        # robust: median then Savitzky-Golay
        from scipy.ndimage import median_filter
        P = median_filter(P, size=(5, 1, 1), mode="nearest")
        P = savgol_filter(P, w, 2, axis=0, mode="nearest")
    out = []
    for p, c in zip(P, C):
        G, _ = cv2.findHomography(ref, p, 0)
        H = c @ G
        out.append(H / H[2, 2])
    return np.array(out)
