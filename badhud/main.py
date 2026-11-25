"""hud <clip>  ->  <name>_HUD.mp4  +  <name>_HUD_report/ (check report)  +  <name>_hud.yaml (edits)."""
import argparse, os, sys, time, pickle, shutil
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hud", description="Game-style HUD for a badminton rally clip")
    ap.add_argument("clip")
    ap.add_argument("-o", "--out", help="output video (default <name>_HUD.mp4 next to the clip)")
    ap.add_argument("--config", help="edit file (default <name>_hud.yaml next to the clip)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached analysis")
    ap.add_argument("--model", default=None, help="YOLO weights (default yolo11m.pt)")
    ap.add_argument("--no-video", action="store_true", help="only analysis + report (skip rendering)")
    a = ap.parse_args(argv)

    from . import prep, people, camera, court, identity, ball, shots, render, report, config as cfgm

    clip = os.path.abspath(a.clip)
    base, _ = os.path.splitext(clip)
    name = os.path.basename(base)
    out = a.out or base + "_HUD.mp4"
    work = base + "_hudwork"
    if a.fresh and os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    cfg_path = a.config or base + "_hud.yaml"
    cfg = cfgm.load(cfg_path)
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.1f}s] {s}", flush=True)

    log("1/10 prepare video + audio")
    video, wav, fps = prep.prepare(clip, work)
    onsets, _ = prep.hit_onsets(wav, fps)
    log(f"     fps={fps:.2f}, audio onsets={len(onsets)}")

    log("3/10 people (YOLO11m + ByteTrack, imgsz 1280)")
    model = a.model or _find_model()
    ppl = people.track_people(video, work, model, log)
    T = len(ppl)

    log("2/10 camera motion (ceiling/wall homography)")
    Ms = camera.estimate_motion(video, ppl, work, log)
    C = camera.cumulative(Ms)

    log("4-5/10 court: first-frame calibration + per-frame tracking")
    first = next(camera.frames(video))
    H0, s0, dbg0 = court.detect_court(first, ppl[0])
    if cfg.get("court_points"):
        H0 = cfgm.court_from_points(cfg["court_points"])
    if H0 is None:
        sys.exit("court not found in the first frame; add court_points to the config")
    log(f"     first frame overlap score {s0:.0f}")
    ct = court.track_court(video, ppl, Ms, work, H0, log)
    Hs = court.smooth_court(ct["H_raw"], C)

    log("6/10 who is who")
    ipath = os.path.join(work, "identity.pkl")
    if os.path.exists(ipath):
        assign = pickle.load(open(ipath, "rb"))
    else:
        assign = identity.assign_all(video, ppl, Hs, cfg["slots"], fps, log)
        pickle.dump(assign, open(ipath, "wb"))
    pos, boxes, held, seen = identity.fill_tracks(assign, fps)

    log("7/10 shuttle")
    cands = ball.detect_candidates(video, C, Hs, ppl, work, log)
    tpath = os.path.join(work, "ball_tracklets.pkl")
    if os.path.exists(tpath):
        trk = pickle.load(open(tpath, "rb"))
    else:
        trk, _ = ball.build_tracklets(cands, C)
        pickle.dump(trk, open(tpath, "wb"))
    path, all_tr = ball.select_path(trk, C, boxes, fps, log)
    bpts = ball.path_points(path, C)

    log("8/10 shots")
    sh = shots.detect(path, all_tr, bpts, C, Hs, pos, boxes, onsets, fps, T, cfg, log)

    log("9/10 render HUD")
    if not a.no_video:
        render.render(video, wav, out, Hs, pos, boxes, held, bpts, sh, cfg, fps, log)

    log("10/10 check report")
    rdir = base + "_HUD_report"
    report.build(rdir, video, out, Hs, ct, pos, boxes, assign, bpts, sh, cands, cfg, cfg_path, fps, dbg0, log)
    cfgm.write_template(cfg_path, cfg, sh)
    log(f"done -> {out}\n       report -> {os.path.join(rdir, 'index.html')}\n       edits  -> {cfg_path}")


def _find_model():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(here, "yolo11m.pt"), "yolo11m.pt"):
        if os.path.exists(p):
            return p
    return "yolo11m.pt"  # ultralytics downloads it


if __name__ == "__main__":
    main()
