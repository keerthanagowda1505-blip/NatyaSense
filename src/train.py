import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import ROOT, load_config


def build(kind, seed):
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000,
                          random_state=seed))
    return HistGradientBoostingClassifier(
        max_iter=150, min_samples_leaf=3, random_state=seed)


def main():
    cfg = load_config()
    seed = cfg["training"]["seed"]
    feat_dir = ROOT / cfg["paths"]["feature_dir"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    report_dir = ROOT / cfg["paths"]["report_dir"]
    models_dir.mkdir(exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    for task in ("mudra", "pose"):
        path = feat_dir / f"{task}.csv"
        if not path.exists():
            print(f"{task}: no features, skipped")
            continue
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c.startswith("f")]
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "reference"]

        kind = cfg["training"][task]["model"]
        model = build(kind, seed)
        model.fit(tr[cols].values, tr["label"].values)
        pred = model.predict(te[cols].values)

        acc = accuracy_score(te["label"], pred)
        labels = sorted(df["label"].unique())
        text = (f"Task: {task}   model: {kind}\n"
                f"Train photos: {len(tr)}   Test photos: {len(te)}\n"
                f"Accuracy: {acc:.2%}\n\n"
                f"{classification_report(te['label'], pred, zero_division=0)}\n"
                f"Confusion matrix (rows = true, cols = predicted)\n"
                f"Labels: {labels}\n"
                f"{confusion_matrix(te['label'], pred, labels=labels)}\n")
        print(text)
        (report_dir / f"{task}.txt").write_text(text, encoding="utf-8")
        joblib.dump({"model": model, "features": cols, "labels": labels},
                    models_dir / f"{task}_model.joblib")


if __name__ == "__main__":
    main()