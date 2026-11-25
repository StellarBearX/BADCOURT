"""Step 2: hand-held camera motion = homography between consecutive frames, estimated from
feature points on the ceiling and walls (above the floor, outside player boxes)."""
import os
import numpy as np
import cv2
from .court import floor_limit


def frames(video):
    cap = cv2.VideoCapture(video)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def estimate_motion(video, people, work, log=print):
    """M[t] maps pixels of frame t-1 to frame t (M[0] = I)."""
    out = os.path.join(work, "camera.npy")
    if os.path.exists(out):
        return np.load(out)
    Ms = []
    prev = None
    sc = 0.5
    for t, f in enumerate(frames(video)):
        g = cv2.cvtColor(cv2.resize(f, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cv2.resize(f, None, fx=sc, fy=sc), cv2.COLOR_BGR2HSV)
        lim = floor_limit(hsv)
        mask = (np.arange(g.shape[0])[:, None] < (lim[None, :] - 10)).astype(np.uint8) * 255
        for b in people[t] if t < len(people) else []:
            x1, y1, x2, y2 = (b[:4] * sc).astype(int)
            mask[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8] = 0
        if prev is None:
            Ms.append(np.eye(3))
        else:
            pg, pmask = prev
            pts = cv2.goodFeaturesToTrack(pg, 600, 0.01, 8, mask=pmask, blockSize=7)
            M = None
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None, winSize=(21, 21), maxLevel=4)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(g, pg, nxt, None, winSize=(21, 21), maxLevel=4)
                fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
                if good.sum() >= 12:
                    M, inl = cv2.findHomography(pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2),
                                                cv2.RANSAC, 1.5)
            if M is None:
                M = Ms[-1] if len(Ms) > 1 else np.eye(3)
            S = np.diag([sc, sc, 1.0])
            Ms.append(np.linalg.inv(S) @ M @ S)
        prev = (g, mask)
        if t % 100 == 0:
            log(f"  camera: frame {t}")
    Ms = np.array(Ms)
    np.save(out, Ms)
    return Ms


def cumulative(Ms):
    """C[t] maps frame-0 pixels to frame-t pixels."""
    C = [np.eye(3)]
    for M in Ms[1:]:
        c = M @ C[-1]
        C.append(c / c[2, 2])
    return np.array(C)
