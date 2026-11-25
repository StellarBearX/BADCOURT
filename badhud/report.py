"""Step 10: check report (HTML + images): court lines on 8 frames, every shot (at the hit and
0.2 s later), long stretches without a shot, identity strip, warnings and how to edit."""
import os, html, json
import numpy as np
import cv2
from .court import draw_court, line_mask
from .camera import frames
from .render import hex2rgb


def _k(s):
    return "---" if s["kmh"] is None else f"{s['kmh']:.0f}"


def _bgr(hexcol):
    r, g, b = hex2rgb(hexcol)
    return (b, g, r)


def _annot(fr, f, Hs, boxes, slots, bpts, hl_slot=None, text=None, court=True):
    im = fr.copy()
    if court:
        draw_court(im, Hs[f], (0, 255, 255), 2)
    for s in range(4):
        b = boxes[s, f]
        if np.isnan(b).any():
            continue
        x1, y1, x2, y2 = b.astype(int)
        th = 6 if s == hl_slot else 2
        cv2.rectangle(im, (x1, y1), (x2, y2), _bgr(slots[s]["color"]), th)
        cv2.putText(im, f"P{s + 1}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, _bgr(slots[s]["color"]), 3)
    # shuttle trail
    trail = [(k, bpts[k]) for k in range(f - 12, f + 1) if k in bpts]
    for k, p in trail:
        cv2.circle(im, tuple(np.int32(p)), 5 if k < f else 14, (0, 0, 255), 2 if k == f else -1)
    future = [(k, bpts[k]) for k in range(f + 1, f + 13) if k in bpts]
    for k, p in future:
        cv2.circle(im, tuple(np.int32(p)), 5, (255, 0, 255), -1)
    if text:
        cv2.rectangle(im, (0, 0), (1080, 70), (0, 0, 0), -1)
        cv2.putText(im, text, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return im


def build(rdir, video, out, Hs, ct, pos, boxes, assign, bpts, sh, cands, cfg, cfg_path, fps, dbg0, log=print):
    os.makedirs(rdir, exist_ok=True)
    for fn in os.listdir(rdir):
        if fn.endswith(".jpg"):
            os.remove(os.path.join(rdir, fn))
    slots = cfg["slots"]
    shots, rallies, warns = sh["shots"], sh["rallies"], sh["warnings"]
    T = len(Hs)
    want = {}

    def need(f, key):
        want.setdefault(int(np.clip(f, 0, T - 1)), []).append(key)

    court_frames = np.linspace(0, T - 1, 8).astype(int)
    for f in court_frames:
        need(f, ("court", f))
    for k, s in enumerate(shots):
        need(s["f"], ("shot", k, 0))
        need(s["f"] + round(0.2 * fps), ("shot", k, 1))
    gaps = []
    for k in range(len(shots) - 1):
        a, b = shots[k], shots[k + 1]
        if b["t"] - a["t"] > 2.4:
            m = int((a["f"] + b["f"]) / 2)
            gaps.append((a["t"], b["t"], m, a["rally"] == b["rally"]))
            need(m, ("gap", len(gaps) - 1))
    id_frames = np.arange(0, T, int(4 * fps))
    for f in id_frames:
        need(f, ("id", f))

    imgs = {}
    for f, fr in enumerate(frames(video)):
        if f not in want:
            continue
        for key in want[f]:
            if key[0] == "court":
                im = fr.copy()
                m, _ = line_mask(fr)
                im[m > 0] = (0.5 * im[m > 0] + [0, 0, 127]).astype(np.uint8)
                draw_court(im, Hs[f], (0, 255, 255), 3)
                info = ct["info"][f]
                cv2.rectangle(im, (0, 0), (1080, 70), (0, 0, 0), -1)
                cv2.putText(im, f"frame {f}  t={f / fps:.2f}s  overlap={info['score']:.0f} ({info['src']})",
                            (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                name = f"court_{f:05d}.jpg"
            elif key[0] == "shot":
                k, ph = key[1], key[2]
                s = shots[k]
                lab = f"#{k + 1} t={f / fps:.2f}s {'HIT' if ph == 0 else '+0.2s'} P{s['slot'] + 1} {_k(s)}km/h [{s['src']}]"
                im = _annot(fr, f, Hs, boxes, slots, bpts, s["slot"], lab, court=False)
                name = f"shot_{k + 1:03d}_{ph}.jpg"
            elif key[0] == "gap":
                g = gaps[key[1]]
                im = _annot(fr, f, Hs, boxes, slots, bpts, None,
                            f"no shot {g[0]:.2f}-{g[1]:.2f}s ({g[1] - g[0]:.1f}s)", court=False)
                name = f"gap_{key[1]:03d}.jpg"
            else:
                im = _annot(fr, f, Hs, boxes, slots, bpts, None, f"t={f / fps:.1f}s identity", court=True)
                name = f"id_{f:05d}.jpg"
            small = cv2.resize(im, (540, 960), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(rdir, name), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
            imgs[key] = name
    # first-frame calibration debug (full res crop of the floor)
    if dbg0 is not None:
        fr = next(frames(video))
        im = fr.copy()
        for x1, y1, x2, y2 in dbg0.get("segs", np.zeros((0, 4))).astype(int):
            cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        draw_court(im, Hs[0], (0, 255, 255), 2)
        if dbg0.get("vp") is not None:
            vp = tuple(np.int32(dbg0["vp"]))
            cv2.circle(im, vp, 12, (255, 0, 255), 3)
        cv2.imwrite(os.path.join(rdir, "calib_first.jpg"), im[900:, :], [cv2.IMWRITE_JPEG_QUALITY, 85])

    names = [s["name"] for s in slots]
    E = html.escape
    rows = []
    for k, s in enumerate(shots):
        w = [x["msg"] for x in warns if x["shot"] == k + 1]
        rows.append(f"<tr class='{'warn' if w else ''}'><td>{k + 1}</td><td>{s['t']:.2f}</td><td>{(s['rally'] or 0) + 1}</td>"
                    f"<td>P{s['slot'] + 1} {E(s['name'])}</td><td>{'ใกล้' if s['team'] == 0 else 'ไกล'}</td>"
                    f"<td>{_k(s)}{'*' if s['est'] else ''}</td><td>{s['D']:.1f}</td><td>{s['T']:.2f}</td><td>{s['src']}</td>"
                    f"<td>{E('; '.join(w))}</td></tr>")
    shot_cards = []
    for k, s in enumerate(shots):
        a, b = imgs.get(("shot", k, 0)), imgs.get(("shot", k, 1))
        shot_cards.append(f"<div class='card'><h4>#{k + 1} · {s['t']:.2f}s · {E(s['name'])} · {_k(s)} km/h · {s['src']}</h4>"
                          f"<img src='{a}'><img src='{b}'></div>")
    gap_cards = [f"<div class='card'><h4>{g[0]:.2f}–{g[1]:.2f}s ({g[1] - g[0]:.1f}s){'' if g[3] else ' · ระหว่างแรลลี'}</h4><img src='{imgs[('gap', i)]}'></div>"
                 for i, g in enumerate(gaps) if ("gap", i) in imgs]
    court_cards = [f"<div class='card'><img src='{imgs[('court', f)]}'></div>" for f in court_frames if ("court", f) in imgs]
    id_cards = [f"<div class='card'><img src='{imgs[('id', f)]}'></div>" for f in id_frames if ("id", f) in imgs]
    rally_txt = ", ".join(f"แรลลี {i + 1}: {a / fps:.2f}–{b / fps:.2f}s" for i, (a, b) in enumerate(rallies))
    warn_li = "".join(f"<li>t={w['t']:.2f}s (ช็อต {w['shot']}): {E(w['msg'])}</li>" for w in warns) or "<li>ไม่มี</li>"
    legend = " ".join(f"<span style='color:{s['color']}'>■ P{i + 1} {E(s['name'])}</span>" for i, s in enumerate(slots))
    doc = f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HUD check report</title><style>
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:16px}}
h1,h2{{color:#39d0ff}} .grid{{display:flex;flex-wrap:wrap;gap:10px}} .card{{background:#161b22;padding:8px;border-radius:8px}}
.card img{{width:270px;margin-right:4px}} .card h4{{margin:2px 0 6px;font-size:14px}}
table{{border-collapse:collapse;font-size:14px}} td,th{{border:1px solid #30363d;padding:4px 8px}} tr.warn{{background:#3d1d1d}}
code,pre{{background:#161b22;padding:2px 6px;border-radius:4px}} .wide img{{width:540px}}
</style></head><body>
<h1>รายงานตรวจ HUD</h1>
<p>วิดีโอ: <code>{E(os.path.basename(out))}</code> · {T} เฟรม @ {fps:.0f} fps · {len(shots)} ช็อต · {rally_txt}</p>
<p>{legend}</p>
<h2>คำเตือน ({len(warns)})</h2><ul>{warn_li}</ul>
<h2>ตารางช็อต</h2>
<table><tr><th>#</th><th>เวลา (s)</th><th>แรลลี</th><th>คนตี</th><th>ฝั่ง</th><th>km/h</th><th>D (m)</th><th>T (s)</th><th>ที่มา</th><th>เตือน</th></tr>{''.join(rows)}</table>
<p>* = ช็อตสุดท้าย ใช้ตำแหน่งลูกตกที่ประมาณจากภาพ · ที่มา: ball = ลูกกลับทิศที่ไม้, audio = เสียงตีตอนมองไม่เห็นลูก, manual = จากไฟล์แก้</p>
<h2>แก้ไข</h2><p>แก้ไฟล์ <code>{E(cfg_path)}</code> แล้วรัน <code>hud &lt;คลิป&gt;</code> ซ้ำ:</p>
<pre>shots_add:
  - {{t: 12.34, player: "{E(names[2])}"}}
shots_remove:
  - 15.20
rally_end: 47.8</pre>
<h2>เส้นคอร์ท 8 เฟรม (เหลือง = โมเดลคอร์ท, แดง = พิกเซลเส้นขาวที่ตรวจเจอ)</h2><div class="grid">{''.join(court_cards)}</div>
<h2>คาลิเบรตเฟรมแรก (แดง = Hough, ม่วง = จุดลู่)</h2><div class="card wide"><img src="calib_first.jpg"></div>
<h2>ทุกช็อต (ซ้าย = ตอนตี, ขวา = 0.2 วิหลังตี · จุดแดง = ลูกก่อนหน้า, ม่วง = ลูกหลังจากนี้)</h2><div class="grid">{''.join(shot_cards)}</div>
<h2>ช่วงที่ไม่มีช็อตนาน (&gt; 2.4 วิ)</h2><div class="grid">{''.join(gap_cards) or 'ไม่มี'}</div>
<h2>ใครเป็นใคร (ทุก 4 วิ)</h2><div class="grid">{''.join(id_cards)}</div>
</body></html>"""
    open(os.path.join(rdir, "index.html"), "w", encoding="utf-8").write(doc)
    json.dump({"shots": shots, "rallies": [[a / fps, b / fps] for a, b in rallies], "warnings": warns},
              open(os.path.join(rdir, "shots.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"     report: {len(imgs)} images")
