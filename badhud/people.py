"""Step 3: YOLO11m + ByteTrack person tracking on every frame (imgsz 1280)."""
import os, pickle
import numpy as np


def track_people(video, work, model_path="yolo11m.pt", log=print):
    out = os.path.join(work, "people.pkl")
    if os.path.exists(out):
        return pickle.load(open(out, "rb"))
    import torch
    from ultralytics import YOLO
    dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    res = []
    for i, r in enumerate(model.track(source=video, stream=True, imgsz=1280, classes=[0], conf=0.15,
                                       iou=0.6, tracker="bytetrack.yaml", device=dev, verbose=False,
                                       persist=True)):
        b = r.boxes
        if b is None or len(b) == 0:
            res.append(np.zeros((0, 6), np.float32))
        else:
            ids = b.id.cpu().numpy() if b.id is not None else -np.ones(len(b))
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], ids[:, None]], 1)
            res.append(arr.astype(np.float32))
        if i % 50 == 0:
            log(f"  people: frame {i}")
            pickle.dump(res, open(out + ".partial", "wb"))
    pickle.dump(res, open(out, "wb"))
    return res
