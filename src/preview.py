import json
import sys
import cv2
from src.config import ROOT, load_config

POSE_LINES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
              (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),
              (27,31),(28,30),(30,32),(28,32)]
HAND_LINES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
              (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
              (13,17),(17,18),(18,19),(19,20),(0,17)]


def draw(task, label, seq):
    cfg = load_config()
    name = f"{task}__{label}__{seq}"
    data = json.loads((ROOT / cfg["paths"]["landmark_dir"] / f"{name}.json")
                      .read_text(encoding="utf-8"))
    img = cv2.imread(str(ROOT / data["meta"]["path"]))
    h, w = img.shape[:2]
    k = 900 / max(h, w)
    img = cv2.resize(img, (int(w * k), int(h * k)))
    h, w = img.shape[:2]

    if data["pose"]:
        pts = [(int(p[0] * w), int(p[1] * h)) for p in data["pose"]]
        for a, b in POSE_LINES:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts[11:]:
            cv2.circle(img, p, 4, (0, 0, 255), -1)

    for hd in data["hands"]:
        pts = [(int(p[0] * w), int(p[1] * h)) for p in hd["points"]]
        for a, b in HAND_LINES:
            cv2.line(img, pts[a], pts[b], (255, 0, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 255, 255), -1)
        cv2.putText(img, hd["side"], pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

    out_dir = ROOT / "outputs" / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.jpg"
    cv2.imwrite(str(out), img)
    print("Saved", out)


if __name__ == "__main__":
    draw(sys.argv[1], sys.argv[2], sys.argv[3])