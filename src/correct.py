import json
import joblib
import numpy as np
from src.config import ROOT, load_config

# Plain-language names for each feature (temporary; the feedback bank replaces these later)
HAND_NAMES = (
    [f"{f} finger {j} bend" for f in ("thumb", "index", "middle", "ring", "little")
     for j in ("lower", "upper")]
    + [f"spread between {a} and {b}" for a, b in
       (("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "little"))]
    + [f"thumb-to-{f} distance" for f in ("index", "middle", "ring", "little")]
    + [f"{f} reach" for f in ("thumb", "index", "middle", "ring", "little")])

POSE_NAMES = (
    ["left elbow bend", "right elbow bend", "left shoulder angle",
     "right shoulder angle", "left hip angle", "right hip angle",
     "left knee bend", "right knee bend", "left shoulder line",
     "right shoulder line"]
    + [f"{p} {ax} position" for p in ("left wrist", "right wrist",
       "left ankle", "right ankle", "head") for ax in ("sideways", "vertical")])


def load(task):
    cfg = load_config()
    bundle = joblib.load(ROOT / cfg["paths"]["models_dir"] / f"{task}_model.joblib")
    bands = json.loads((ROOT / "models" / "bands.json").read_text(encoding="utf-8"))
    return cfg, bundle, bands.get(task, {})


def check(task, features, level="advanced"):
    cfg, bundle, bands = load(task)
    inf = cfg["inference"]
    k_in, k_out = cfg["bands"]["inner_k"], cfg["bands"]["outer_k"]
    mult = cfg["levels"][level]

    model, cols = bundle["model"], bundle["features"]
    x = np.array(features, dtype=float).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    label = str(model.classes_[int(proba.argmax())])
    conf = float(proba.max())

    result = {"label": label, "confidence": round(conf, 3),
              "status": "ok", "messages": []}
    if conf < inf["tau"]:
        result["status"] = "not_sure"
        result["messages"] = ["Not sure which one this is. Try again, facing the camera."]
        return result
    if label not in bands:
        result["status"] = "no_reference"
        return result

    names = HAND_NAMES if task == "mudra" else POSE_NAMES
    issues = []
    for i, c in enumerate(cols):
        b = bands[label][c]
        z = abs(features[i] - b["mean"]) / (b["std"] * mult)
        if z > k_in:
            issues.append((z, names[i], features[i] > b["mean"],
                           "major" if z > k_out else "minor"))
    issues.sort(reverse=True)
    for z, name, high, sev in issues[:inf["max_messages"]]:
        direction = "too high / too wide" if high else "too low / too narrow"
        result["messages"].append(f"{name}: {direction} ({sev})")
    result["score"] = round(max(0, 100 - 8 * sum(z for z, *_ in issues)), 1)
    result["tier"] = ("excellent" if not issues else
                      "good" if issues[0][3] == "minor" else "needs work")
    return result


if __name__ == "__main__":
    # Test on one reference photo per task
    import pandas as pd
    for task, label, seq in (("mudra", "alapadma", 17), ("pose", "kali_devi", 21)):
        cfg = load_config()
        df = pd.read_csv(ROOT / cfg["paths"]["feature_dir"] / f"{task}.csv")
        row = df[(df["label"] == label) & (df["seq"] == seq)]
        if row.empty:
            print(task, label, seq, "- no features for this photo")
            continue
        cols = [c for c in df.columns if c.startswith("f")]
        print(task, label, seq, "->", check(task, row[cols].iloc[0].tolist()))