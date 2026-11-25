"""Step 1: normalise video to 1080x1920, extract audio, find hit onsets (2-8 kHz)."""
import json, os, subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

W, H = 1080, 1920


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,nb_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return float(n) / float(d)


def prepare(src, work):
    vid = os.path.join(work, "video.mp4")
    wav = os.path.join(work, "audio.wav")
    if not os.path.exists(vid):
        run(["ffmpeg", "-y", "-i", src, "-vf",
             f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2",
             "-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", vid])
    has_audio = True
    if not os.path.exists(wav):
        try:
            run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "32000", wav])
        except subprocess.CalledProcessError:
            has_audio = False
    return vid, (wav if has_audio and os.path.exists(wav) else None), probe(vid)


def hit_onsets(wav, fps, min_gap=0.25):
    """Onset times (s) of transient energy in the 2-8 kHz band (racket hits)."""
    if wav is None:
        return [], np.zeros(0)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    x /= (np.abs(x).max() + 1e-9)
    sos = butter(6, [2000, min(8000, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    hop = int(sr * 0.005)
    env = np.sqrt(np.convolve(y ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    env_db = 20 * np.log10(env + 1e-6)
    # onset strength: rise vs. recent past
    k = 10  # 50 ms
    past = np.array([env_db[max(0, i - k):i].mean() if i > 0 else env_db[0] for i in range(len(env_db))])
    flux = np.maximum(env_db - past, 0)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-6
    thr = med + 8 * mad
    ons = []
    i = 0
    while i < len(flux):
        if flux[i] > thr and flux[i] >= flux[max(0, i - 3):i + 4].max():
            t = i * hop / sr
            if not ons or t - ons[-1][0] > min_gap:
                ons.append((t, float(flux[i])))
            elif flux[i] > ons[-1][1]:
                ons[-1] = (t, float(flux[i]))
        i += 1
    return ons, env_db
