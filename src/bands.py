import json
import pandas as pd
from src.config import ROOT, load_config


def main():
    cfg = load_config()
    b = cfg["bands"]
    feat_dir = ROOT / cfg["paths"]["feature_dir"]
    out = {}

    for task in ("mudra", "pose"):
        path = feat_dir / f"{task}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c.startswith("f")]
        # reference photos are the "correct" examples
        ref = df[df["split"] == "reference"]
        out[task] = {}
        for label, g in ref.groupby("label"):
            if len(g) < b["min_frames"]:
                print(f"{task}/{label}: only {len(g)} photos, skipped")
                continue
            mean, std = g[cols].mean(), g[cols].std().fillna(0.05)
            out[task][label] = {
                c: {"mean": float(mean[c]), "std": float(max(std[c], 0.05))}
                for c in cols}
            print(f"{task}/{label}: bands from {len(g)} photos")

    dest = ROOT / "models" / "bands.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("Saved", dest)


if __name__ == "__main__":
    main()