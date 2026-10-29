from src.config import load_config
from src.extract import make_detectors, load_image, detect
from src.features import hand_features, pose_features
from src.correct import check

_cfg = load_config()
_pose, _hands = make_detectors(_cfg)
_m = _cfg["mediapipe"]


def analyze(path, task):
    """task is 'mudra' or 'pose'. Returns the check() result."""
    rgb = load_image(path, _m["max_image_side"])
    if rgb is None:
        return {"status": "no_detection", "messages": ["Could not read this photo."]}
    d = detect(rgb, _pose, _hands, _m["swap_handedness"], _m)
    if task == "mudra":
        if not d["hands"]:
            return {"status": "no_detection",
                    "messages": ["No hand found. Show your hand clearly in good light."]}
        best = max(d["hands"], key=lambda h: h["score"])
        feats = hand_features(best["points"])
    else:
        if not d["pose"]:
            return {"status": "no_detection",
                    "messages": ["No body found. Stand fully in the frame."]}
        feats = pose_features(d["pose"])
    r = check(task, feats)
    if "score" in r:  # make the tier agree with the score
        r["tier"] = ("excellent" if r["score"] >= 85 else
                     "good" if r["score"] >= 60 else "needs work")
    return r


if __name__ == "__main__":
    import sys
    print(analyze(sys.argv[1], sys.argv[2]))