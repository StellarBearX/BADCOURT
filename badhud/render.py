"""Step 9: draw the HUD with Pillow (Thai font) and put the original audio back."""
import os, subprocess, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from .court import project, CW, CL, NET_Y
from .camera import frames

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
CYAN = (0, 229, 255)
WHITE = (255, 255, 255)
PANEL = (8, 14, 28, 178)


def font(size, weight=700):
    for p in (os.path.join(FONT_DIR, f"Kanit-{weight}.ttf"),
              "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------ per-frame state
def player_stats(pos, held, fps, rallies=None):
    """Distance run (m) and stamina (%) per player per frame."""
    from scipy.ndimage import uniform_filter1d
    P, T = pos.shape[:2]
    # far players are ~12 m away: 1 px of jitter is several cm, so smooth before integrating
    pos = uniform_filter1d(np.nan_to_num(pos, nan=0.0), int(0.6 * fps) | 1, axis=1, mode="nearest")
    dist = np.zeros((P, T))
    sta = np.full((P, T), 100.0)
    live = np.ones(T, bool)
    if rallies:
        live[:] = False
        for a, b in rallies:
            live[int(max(0, a)):int(min(T, b + 1))] = True
    for s in range(P):
        d, st = 0.0, 100.0
        for t in range(1, T):
            if not live[t]:  # between rallies: no distance, recover
                st = float(min(100.0, st + 2.0 / fps))
                dist[s, t], sta[s, t] = d, st
                continue
            if np.isnan(pos[s, t]).any() or np.isnan(pos[s, t - 1]).any():
                dist[s, t], sta[s, t] = d, st
                continue
            step = float(np.linalg.norm(pos[s, t] - pos[s, t - 1]))
            v = step * fps
            if v > 9:  # identity glitch, not running
                step, v = 0.0, 0.0
            if v > 0.6:
                d += step
                st -= 0.55 * step + (0.9 / fps if v > 3.0 else 0)
            else:
                st += 0.5 / fps
            st = float(np.clip(st, 8, 100))
            dist[s, t], sta[s, t] = d, st
    return dist, sta


def ball_track(bpts, T, max_gap=4):
    """Observed shuttle points plus linear fill over short gaps."""
    out = {}
    ks = sorted(bpts)
    for a, b in zip(ks[:-1], ks[1:]):
        out[a] = bpts[a]
        if 1 < b - a <= max_gap:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                out[t] = (1 - w) * np.asarray(bpts[a]) + w * np.asarray(bpts[b])
    if ks:
        out[ks[-1]] = bpts[ks[-1]]
    return out


def rally_state(f, shots, rallies, fps):
    r = None
    for k, (a, b) in enumerate(rallies):
        if a <= f <= b + 1.5 * fps:
            r = k
    if r is None:
        prev = [k for k, (a, b) in enumerate(rallies) if b < f]
        if not prev:
            return {"time": 0.0, "count": 0, "speed": None, "live": False, "end": False, "last": None}
        r = prev[-1]
    a, b = rallies[r]
    rs = [s for s in shots if s["rally"] == r and s["f"] <= f]
    live = a <= f <= b
    return {"time": (min(f, b) - a) / fps if f >= a else 0.0, "count": len(rs),
            "speed": rs[-1]["kmh"] if rs else None, "live": live, "end": f > b,
            "last": rs[-1] if rs else None, "rally": r, "n_rallies": len(rallies)}


# ------------------------------------------------------------------ drawing helpers
def court_ellipse(H, c, r, n=48, phase=0.0):
    a = np.linspace(0, 2 * np.pi, n) + phase
    pts = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]
    p, z = project(H, pts)
    return p, (z > 0).all()


def draw_ring(d, H, c, color, t, strong=False):
    col = hex2rgb(color)
    outer, ok = court_ellipse(H, c, 0.55)
    if not ok:
        return
    inner, _ = court_ellipse(H, c, 0.42)
    d.polygon([tuple(p) for p in outer], fill=col + (50,))
    d.line([tuple(p) for p in np.r_[outer, outer[:1]]], fill=col + (235,), width=5 if strong else 4)
    # rotating dashed inner ring
    n = 24
    for k in range(0, n, 2):
        a0 = 2 * np.pi * k / n + t * 1.6
        seg = np.linspace(a0, a0 + 2 * np.pi / n, 5)
        pts = np.c_[c[0] + 0.42 * np.cos(seg), c[1] + 0.42 * np.sin(seg)]
        p, _ = project(H, pts)
        d.line([tuple(q) for q in p], fill=col + (200,), width=3)
    # pointer ticks (front/back/left/right)
    for ang in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        pts = np.array([[c[0] + 0.58 * np.cos(ang), c[1] + 0.58 * np.sin(ang)],
                        [c[0] + 0.72 * np.cos(ang), c[1] + 0.72 * np.sin(ang)]])
        p, _ = project(H, pts)
        d.line([tuple(q) for q in p], fill=col + (235,), width=4)


def text_w(d, s, f):
    b = d.textbbox((0, 0), s, font=f)
    return b[2] - b[0], b[3] - b[1]


def tag_rect(d, box, name, fnt):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    w, h = text_w(d, name, fnt)
    W, Hh = w + 2 * 14 + 10, 50
    return [cx - W / 2, y1 - Hh - 26, cx + W / 2, y1 - 26], cx


def place_tags(d, items, fnt):
    """items: (slot, box, name). Returns slot -> (rect, anchor_x, anchor_y); tags pushed up
    when they would overlap."""
    placed = {}
    rects = []
    for s, box, name in sorted(items, key=lambda it: -it[1][1]):
        r, cx = tag_rect(d, box, name, fnt)
        r[0] = max(4, r[0]); r[2] = r[0] + (r[2] - r[0])
        if r[2] > 1076:
            r[0], r[2] = 1076 - (r[2] - r[0]), 1076
        moved = True
        while moved:
            moved = False
            for q in rects:
                if r[0] < q[2] + 4 and r[2] > q[0] - 4 and r[1] < q[3] + 4 and r[3] > q[1] - 4:
                    dy = r[3] - q[1] + 6
                    r[1] -= dy; r[3] -= dy
                    moved = True
        rects.append(r)
        placed[s] = (r, cx, box[1])
    return placed


def draw_tag(d, rect, cx, ytop, name, color, fnt, active):
    col = hex2rgb(color)
    left, top, right, bot = rect
    d.rounded_rectangle([left, top, right, bot], radius=10, fill=PANEL[:3] + (200,),
                        outline=col + (255,), width=3 if active else 2)
    d.rectangle([left + 3, top + 6, left + 9, bot - 6], fill=col + (255,))
    d.text((left + 22, (top + bot) / 2), name, font=fnt, fill=WHITE + (255,), anchor="lm")
    px = min(max(cx, left + 14), right - 14)
    if ytop - bot > 30:  # pushed up: leader line to the head
        d.line([(px, bot), (cx, ytop - 8)], fill=col + (200,), width=2)
    d.polygon([(px - 11, bot), (px + 11, bot), (px, bot + 14)], fill=col + (255,))


def draw_lock(d, p, t, label, sub, fnt, fnt2):
    x, y = p
    s = 34 + 5 * math.sin(t * 9)
    L = 14
    col = (255, 60, 60, 255)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = x + sx * s, y + sy * s
            d.line([(cx, cy), (cx - sx * L, cy)], fill=col, width=4)
            d.line([(cx, cy), (cx, cy - sy * L)], fill=col, width=4)
    d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=2)
    d.line([(x - s - 18, y), (x - s - 6, y)], fill=col, width=2)
    d.line([(x + s + 6, y), (x + s + 18, y)], fill=col, width=2)
    # label box
    bx, by = x + s + 14, y - s - 44
    w1, _ = text_w(d, label, fnt)
    w2, _ = text_w(d, sub, fnt2) if sub else (0, 0)
    W = max(w1, w2) + 24
    Hh = 36 + (32 if sub else 0)
    if bx + W > 1070:
        bx = x - s - 14 - W
    by = max(160, by)
    d.rectangle([bx, by, bx + W, by + Hh], fill=(40, 0, 0, 170), outline=col, width=2)
    d.text((bx + 12, by + 18), label, font=fnt, fill=(255, 90, 90, 255), anchor="lm")
    if sub:
        d.text((bx + 12, by + 50), sub, font=fnt2, fill=WHITE + (255,), anchor="lm")


def skew_panel(d, x, y, w, h, fill, outline=None, k=18):
    pts = [(x + k, y), (x + w, y), (x + w - k, y + h), (x, y + h)]
    d.polygon(pts, fill=fill)
    if outline:
        d.line(pts + [pts[0]], fill=outline, width=2)


def draw_topbar(d, st, F):
    items = [("RALLY TIME", f"{int(st['time'] // 60):02d}:{st['time'] % 60:04.1f}"),
             ("RALLY COUNT", f"{st['count']:02d}"),
             ("SHOT SPEED", f"{st['speed']:.0f}" if st["speed"] else "---")]
    W = 330
    x0 = (1080 - 3 * W - 2 * 12) / 2
    for i, (lab, val) in enumerate(items):
        x = x0 + i * (W + 12)
        skew_panel(d, x, 40, W, 118, PANEL, CYAN + (220,))
        d.text((x + W / 2, 66), lab, font=F["lab"], fill=CYAN + (255,), anchor="mm")
        if i == 2 and st["speed"]:
            vw, _ = text_w(d, val, F["big"])
            d.text((x + W / 2 - 30, 116), val, font=F["big"], fill=WHITE + (255,), anchor="mm")
            d.text((x + W / 2 - 30 + vw / 2 + 8, 126), "km/h", font=F["small"], fill=CYAN + (255,), anchor="lm")
        else:
            d.text((x + W / 2, 116), val, font=F["big"], fill=WHITE + (255,), anchor="mm")
    if st.get("n_rallies", 1) > 1 and st.get("rally") is not None:
        d.text((540, 178), f"RALLY {st['rally'] + 1}/{st['n_rallies']}", font=F["small"],
               fill=CYAN + (230,), anchor="mm")
    if st["end"]:
        d.text((540, 214), "RALLY END", font=F["lab"], fill=(255, 220, 80, 255), anchor="mm")


def draw_players_panel(d, slots, dist, sta, f, F, active_slot):
    x, y, w, h = 20, 1530, 520, 360
    d.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=PANEL, outline=CYAN + (200,), width=2)
    d.text((x + 20, y + 26), "PLAYERS", font=F["lab"], fill=CYAN + (255,), anchor="lm")
    d.text((x + w - 20, y + 26), "STAMINA · RUN", font=F["small"], fill=(150, 200, 220, 255), anchor="rm")
    for s, p in enumerate(slots):
        ry = y + 62 + s * 74
        col = hex2rgb(p["color"])
        if s == 2:
            d.line([(x + 16, ry - 8), (x + w - 16, ry - 8)], fill=(80, 110, 140, 200), width=1)
        d.rounded_rectangle([x + 14, ry + 10, x + 34, ry + 50], radius=5, fill=col + (255,))
        d.text((x + 24, ry + 30), str(s + 1), font=F["tiny"], fill=(10, 15, 30, 255), anchor="mm")
        nm = p["name"]
        d.text((x + 46, ry + 18), nm, font=F["name"], fill=WHITE + (255,) if s != active_slot else col + (255,), anchor="lm")
        side = "NEAR" if p["side"] == 0 else "FAR"
        d.text((x + 46, ry + 46), side, font=F["tiny"], fill=(140, 170, 190, 255), anchor="lm")
        # stamina bar
        bx, bw = x + 190, 210
        v = sta[s, f] / 100.0
        d.rectangle([bx, ry + 14, bx + bw, ry + 34], fill=(30, 40, 60, 255))
        bc = (80, 255, 120) if v > 0.6 else (255, 210, 60) if v > 0.3 else (255, 70, 70)
        d.rectangle([bx, ry + 14, bx + bw * v, ry + 34], fill=bc + (255,))
        for k in range(1, 10):
            d.line([(bx + bw * k / 10, ry + 14), (bx + bw * k / 10, ry + 34)], fill=(8, 14, 28, 255), width=2)
        d.text((bx, ry + 48), f"{sta[s, f]:.0f}%", font=F["tiny"], fill=(200, 220, 235, 255), anchor="lm")
        d.text((x + w - 18, ry + 24), f"{dist[s, f]:.1f} m", font=F["val"], fill=WHITE + (255,), anchor="rm")


def draw_map(d, pos, f, slots, st, ball_c, F):
    x, y, w, h = 560, 1530, 500, 360
    d.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=PANEL, outline=CYAN + (200,), width=2)
    d.text((x + 20, y + 26), "COURT MAP", font=F["lab"], fill=CYAN + (255,), anchor="lm")
    # landscape mini court: court length along x (far team left, near team right)
    sc = 30.0
    cx0, cy0 = x + (w - CL * sc) / 2, y + 60 + (h - 70 - CW * sc) / 2

    def m(c):  # court metres -> map px  (y along court -> map x)
        return (cx0 + (CL - c[1]) * sc, cy0 + c[0] * sc)

    d.rectangle([m((0, CL)), m((CW, 0))], fill=(20, 70, 110, 220), outline=(220, 240, 255, 255), width=2)
    lc = (220, 240, 255, 200)
    for yy in (0.76, 4.72, 8.68, 12.64):
        d.line([m((0, yy)), m((CW, yy))], fill=lc, width=1)
    for xx in (0.46, 5.64):
        d.line([m((xx, 0)), m((xx, CL))], fill=lc, width=1)
    d.line([m((3.05, 0)), m((3.05, 4.72))], fill=lc, width=1)
    d.line([m((3.05, 8.68)), m((3.05, CL))], fill=lc, width=1)
    d.line([m((-0.3, NET_Y)), m((CW + 0.3, NET_Y))], fill=(255, 255, 255, 255), width=4)
    # last shot arrow
    ls = st.get("last")
    if ls is not None and st["live"]:
        a, b = m(ls["from"]), m(ls["target"])
        d.line([a, b], fill=(255, 80, 80, 220), width=3)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        d.polygon([b, (b[0] - 14 * math.cos(ang - 0.4), b[1] - 14 * math.sin(ang - 0.4)),
                   (b[0] - 14 * math.cos(ang + 0.4), b[1] - 14 * math.sin(ang + 0.4))], fill=(255, 80, 80, 230))
    for s, p in enumerate(slots):
        c = pos[s, f]
        if np.isnan(c).any():
            continue
        c = (float(np.clip(c[0], -1, CW + 1)), float(np.clip(c[1], -1, CL + 1)))
        px, py = m(c)
        col = hex2rgb(p["color"])
        d.ellipse([px - 14, py - 14, px + 14, py + 14], fill=col + (255,), outline=(255, 255, 255, 255), width=2)
        d.text((px, py), str(s + 1), font=F["tiny"], fill=(10, 15, 30, 255), anchor="mm")
    d.text((m((CW / 2, CL))[0], y + h - 22), "FAR", font=F["tiny"], fill=(150, 200, 220, 255), anchor="mm")
    d.text((m((CW / 2, 0))[0], y + h - 22), "NEAR", font=F["tiny"], fill=(150, 200, 220, 255), anchor="mm")


# ------------------------------------------------------------------ main loop
def render(video, wav, out, Hs, pos, boxes, held, bpts, sh, cfg, fps, log=print, frame_range=None):
    slots = cfg["slots"]
    shots, rallies = sh["shots"], sh["rallies"]
    T = len(Hs)
    dist, sta = player_stats(pos, held, fps, rallies)
    bt = ball_track(bpts, T)
    F = {"big": font(54), "lab": font(26, 600), "small": font(24, 500), "tiny": font(20, 500),
         "name": font(30, 600), "val": font(30, 600), "tag": font(30, 600), "lock": font(26, 700),
         "lock2": font(24, 600)}
    tmp = out + ".video.mp4"
    W, Hh = 1080, 1920
    proc = None
    if frame_range is None:
        proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{W}x{Hh}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "18",
                                 "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    shot_frames = np.array([s["f"] for s in shots]) if shots else np.zeros(0)
    imgs = {}
    for f, fr in enumerate(frames(video)):
        if frame_range is not None and f not in frame_range:
            continue
        img = draw_frame(fr, f, Hs, pos, boxes, held, bt, shots, rallies, slots, dist, sta, F, fps, shot_frames)
        if proc is not None:
            proc.stdin.write(img.tobytes())
        else:
            imgs[f] = img
        if f % 100 == 0:
            log(f"  render: frame {f}/{T}")
    if proc is None:
        return imgs
    proc.stdin.close()
    proc.wait()
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", tmp]
    if wav:
        cmd += ["-i", wav, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    os.remove(tmp)


def draw_frame(fr, f, Hs, pos, boxes, held, bt, shots, rallies, slots, dist, sta, F, fps, shot_frames):
    t = f / fps
    base = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    world = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(world)
    st = rally_state(f, shots, rallies, fps)
    last = st["last"]
    active = last["slot"] if (last is not None and (f - last["f"]) < 0.6 * fps) else None
    H = Hs[f]
    # rings under feet (on the court plane)
    order = sorted(range(4), key=lambda s: -np.nan_to_num(pos[s, f, 1], nan=99))  # far first
    for s in order:
        if np.isnan(pos[s, f]).any() or held[s, f]:
            continue
        draw_ring(d, H, pos[s, f], slots[s]["color"], t, strong=(s == active))
    glow = world.resize((540, 960)).filter(ImageFilter.GaussianBlur(6)).resize(base.size)
    base = Image.alpha_composite(base, glow)
    base = Image.alpha_composite(base, world)
    hud = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(hud)
    items = [(s, boxes[s, f], slots[s]["name"]) for s in order
             if not (np.isnan(boxes[s, f]).any() or held[s, f])]
    for s, (r, cx, yt) in place_tags(d, items, F["tag"]).items():
        draw_tag(d, r, cx, yt, slots[s]["name"], slots[s]["color"], F["tag"], s == active)
    # LOCK ON
    if f in bt:
        cur = [x for x in shots if x["f"] <= f]
        sub = None
        if cur and st["live"]:
            c = cur[-1]
            sub = f"{c['name']} · {c['kmh']:.0f} km/h" if c['kmh'] is not None else c['name']
        draw_lock(d, bt[f], t, "LOCK ON", sub, F["lock"], F["lock2"])
    draw_topbar(d, st, F)
    draw_players_panel(d, slots, dist, sta, f, F, active)
    draw_map(d, pos, f, slots, st, None, F)
    base = Image.alpha_composite(base, hud)
    return base.convert("RGB")
