import csv
import json
import numpy as np
from src.config import ROOT, load_config

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20)]
TIPS = [4, 8, 12, 16, 20]


def angle(a, b, c):
    """Angle at point b, in radians."""
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    return float(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1)))


def hand_features(points):
    p = np.array(points)[:, :3]
    scale = np.linalg.norm(p[9] - p[0]) + 1e-9      # wrist to middle knuckle
    f = []
    for a, b, c, d in FINGERS:                       # 2 bend angles per finger
        f.append(angle(p[a], p[b], p[c]))
        f.append(angle(p[b], p[c], p[d]))
    for i in range(4):                               # spread between fingertips
        f.append(angle(p[TIPS[i]], p[0], p[TIPS[i + 1]]))
    for t in TIPS[1:]:                               # thumb tip to other tips
        f.append(np.linalg.norm(p[4] - p[t]) / scale)
    for t in TIPS:                                   # tip distance from wrist
        f.append(np.linalg.norm(p[t] - p[0]) / scale)
    return f


POSE_ANGLES = [(11, 13, 15), (12, 14, 16), (13, 11, 23), (14, 12, 24),
               (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28),
               (13, 11, 12), (14, 12, 11)]


def pose_features(points):
    p = np.array(points)[:, :3]
    f = [angle(p[a], p[b], p[c]) for a, b, c in POSE_ANGLES]
    mid_hip = (p[23] + p[24]) / 2
    scale = np.linalg.norm((p[11] + p[12]) / 2 - mid_hip) + 1e-9
    for i in (15, 16, 27, 28, 0):                    # key points vs hip centre
        v = (p[i] - mid_hip) / scale
        f.extend([float(v[0]), float(v[1])])
    return f


def main():
    cfg = load_config()
    lm_dir = ROOT / cfg["paths"]["landmark_dir"]
    out_dir = ROOT / cfg["paths"]["feature_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0

    for f in sorted(lm_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d["meta"]
        if m["task"] == "mudra" and d["hands"]:
            best = max(d["hands"], key=lambda h: h["score"])
            feat = hand_features(best["points"])
        elif m["task"] == "pose" and d["pose"]:
            feat = pose_features(d["pose"])
        else:
            if m["task"] != "padabheda":
                skipped += 1
            continue
        rows.append([m["task"], m["label"], m["seq"], m["split"]] + feat)

    for task in ("mudra", "pose"):
        sub = [r for r in rows if r[0] == task]
        if not sub:
            continue
        with open(out_dir / f"{task}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            n = len(sub[0]) - 4
            w.writerow(["task", "label", "seq", "split"] +
                       [f"f{i}" for i in range(n)])
            w.writerows(sub)
        print(f"{task}: {len(sub)} rows, {len(sub[0]) - 4} features")
    print(f"Skipped (nothing detected): {skipped}")


if __name__ == "__main__":
    main()