"""Edit file (<name>_hud.yaml): player names, add/remove shots, rally start/end."""
import os
import yaml

DEFAULT = {
    "players": [
        {"name": "พี", "side": "near", "shirt": "dark", "color": "#36C8FF"},
        {"name": "ซี", "side": "near", "shirt": "light", "color": "#7CFF6B"},
        {"name": "เอิร์ท", "side": "far", "shoes": "dark", "color": "#FF5A5A"},
        {"name": "ผู้เล่น 4", "side": "far", "shoes": "white", "color": "#FFB238"},
    ],
    "shots_add": [],
    "shots_remove": [],
    "rally_start": None,
    "rally_end": None,
    "rallies": None,
    "court_points": None,
}

TEMPLATE = """# ---- badhud edit file -------------------------------------------------------
# แก้ไฟล์นี้แล้วรัน  hud <คลิป>  ซ้ำ (ใช้ผลวิเคราะห์ที่เก็บไว้ จึงเร็ว)
#
# players: ชื่อ/ฝั่ง/ลักษณะ  (side: near = ฝั่งใกล้กล้อง, far = ฝั่งตรงข้าม)
#   near แยกกันด้วย shirt: dark | light   ,  far แยกกันด้วย shoes: dark | white
#
# shots_add:    เพิ่มช็อตที่ตรวจไม่เจอ   - {{t: 12.34, player: "เอิร์ท"}}   (player ไม่ใส่ก็ได้)
#               player ใส่เป็นชื่อ หรือ near / far (ให้เลือกคนในทีมที่ใกล้ลูกเอง)
#               ถ้าใส่เวลาเดียวกับช็อตที่ตรวจเจอ (ห่างไม่เกิน 0.2 วิ) = แก้คนตีของช็อตนั้น
# shots_remove: ลบช็อตที่ผิด (เวลาเป็นวินาที ลบช็อตที่ใกล้ที่สุดภายใน 0.2 วิ)   - 12.30
# rally_start / rally_end: เวลาเริ่ม/จบแรลลี (วินาที) ถ้าไม่ใส่ = หาอัตโนมัติ
# rallies: ถ้าคลิปมีหลายแรลลี ใส่เป็นช่วง  - [0.0, 10.4]   - [14.0, 21.0]
#
# ช็อตที่ตรวจเจอรอบล่าสุด (ดูภาพใน report):
{shots}
# ------------------------------------------------------------------------------
"""


def load(path):
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT.items()}
    if os.path.exists(path):
        user = yaml.safe_load(open(path, encoding="utf-8")) or {}
        for k, v in user.items():
            if v is not None or k in ("rally_start", "rally_end", "rallies"):
                cfg[k] = v
    sides = {"near": 0, "far": 1}
    cfg["slots"] = []
    for p in cfg["players"]:
        s = dict(p)
        s["side"] = sides.get(p.get("side", "near"), 0)
        cfg["slots"].append(s)
    # order slots: near first
    return cfg


def write_template(path, cfg, shots):
    """(Re)write the edit file: documentation header + the current values as YAML.
    Values the user set are kept (they were loaded into cfg)."""
    lines = []
    for i, s in enumerate(shots["shots"]):
        lines.append(f"#   {i + 1:2d}. t={s['t']:6.2f}s  {s['name']:<10s} {(format(s['kmh'], '5.0f') if s['kmh'] is not None else '  ---')} km/h  ({s['src']})")
    head = TEMPLATE.format(shots="\n".join(lines) if lines else "#   (none)")
    body = {k: cfg.get(k, DEFAULT[k]) for k in DEFAULT}
    open(path, "w", encoding="utf-8").write(
        head + yaml.safe_dump(body, allow_unicode=True, sort_keys=False, default_flow_style=None, width=120))


def court_from_points(pts):
    """Manual fallback: court_points: [[x_m, y_m, px, py], ...] (>= 4)."""
    import numpy as np, cv2
    a = np.array(pts, float)
    H, _ = cv2.findHomography(a[:, :2], a[:, 2:4], 0)
    return H
