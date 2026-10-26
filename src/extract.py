import csv
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src.config import ROOT, load_config


def make_detectors(cfg):
    m = cfg["mediapipe"]
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ROOT / m["pose_model"])),
            min_pose_detection_confidence=m["min_pose_confidence"],
            num_poses=1))
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ROOT / m["hand_model"])),
            min_hand_detection_confidence=m["min_hand_confidence"],
            num_hands=2))
    return pose, hands


def load_image(path, max_side):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        k = max_side / s
        img = cv2.resize(img, (int(w * k), int(h * k)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _hands_on(rgb, hands, swap):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hr = hands.detect(mp_img)
    res = []
    for lm, hd in zip(hr.hand_landmarks, hr.handedness):
        side = hd[0].category_name
        if swap:  # our photos are NOT mirrored
            side = "Left" if side == "Right" else "Right"
        res.append({"side": side, "score": hd[0].score,
                    "points": [[p.x, p.y, p.z] for p in lm]})
    return res


def _refine(rgb, hand, hands, swap, cfg_m):
    """Crop a square around the hand, enlarge it, detect again."""
    h, w = rgb.shape[:2]
    xs = [p[0] * w for p in hand["points"]]
    ys = [p[1] * h for p in hand["points"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.8
    side = max(side, cfg_m["roi_min_px"])
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return hand
    k = cfg_m["roi_upscale_to"] * 2 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=k, fy=k)
    new = _hands_on(np.ascontiguousarray(crop), hands, swap)
    if not new:
        return hand
    best = max(new, key=lambda r: r["score"])
    ch, cw = crop.shape[:2]
    best["points"] = [[(x0 + p[0] * (x1 - x0)) / w,
                       (y0 + p[1] * (y1 - y0)) / h, p[2]]
                      for p in best["points"]]
    best["refined"] = True
    return best


def detect(rgb, pose, hands, swap, cfg_m=None):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    out = {"pose": None, "hands": []}

    pr = pose.detect(mp_img)
    if pr.pose_landmarks:
        out["pose"] = [[p.x, p.y, p.z, p.visibility]
                       for p in pr.pose_landmarks[0]]

    found = _hands_on(rgb, hands, swap)
    if not found:  # retry: pad the image and enlarge it
        padded = cv2.copyMakeBorder(rgb, 80, 80, 80, 80,
                                    cv2.BORDER_REPLICATE)
        padded = cv2.resize(padded, None, fx=1.5, fy=1.5)
        ph, pw = padded.shape[:2]
        h, w = rgb.shape[:2]
        for r in _hands_on(np.ascontiguousarray(padded), hands, swap):
            r["points"] = [[(p[0] * pw / 1.5 - 80) / w,
                            (p[1] * ph / 1.5 - 80) / h, p[2]]
                           for p in r["points"]]
            r["retry"] = True
            found.append(r)
    if cfg_m:
        found = [_refine(rgb, hd, hands, swap, cfg_m) for hd in found]
    out["hands"] = found
    return out


def main():
    cfg = load_config()
    pose, hands = make_detectors(cfg)
    m = cfg["mediapipe"]
    out_dir = ROOT / cfg["paths"]["landmark_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / cfg["paths"]["manifest"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = fail = 0
    for i, r in enumerate(rows, 1):
        rgb = load_image(ROOT / r["path"], m["max_image_side"])
        if rgb is None:
            print("Cannot read", r["path"])
            fail += 1
            continue
        res = detect(rgb, pose, hands, m["swap_handedness"],m)
        res["meta"] = r
        name = f'{r["task"]}__{r["label"]}__{r["seq"]}.json'
        (out_dir / name).write_text(json.dumps(res), encoding="utf-8")

        found = []
        if res["pose"]:
            found.append("pose")
        if res["hands"]:
            found.append(f'{len(res["hands"])} hand(s)')
        print(f'[{i}/{len(rows)}] {name}: {", ".join(found) or "NOTHING FOUND"}')
        ok += 1

    print(f"\nDone. Saved {ok} files, {fail} unreadable.")


if __name__ == "__main__":
    main()