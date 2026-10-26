import csv
import re
from collections import Counter
from src.config import ROOT, load_config, normalise

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Photos numbered up to this value are used for training.
# Higher numbers are reference/test photos.
TRAIN_MAX = {"mudra": 16, "padabheda": 16, "pose": 20}

EXPECTED = {
    "mudra": ["kapittha", "katakamukha", "alapadma", "chatura", "bhramara",
              "hamsasya", "hamsapaksha", "sandamsha", "karkata",
              "tripataka", "ardhapataka", "mayura"],
    "padabheda": ["hayatham", "anchita", "kunchita", "sampada", "ekapada"],
    "pose": ["lakshmi_devi", "krishna", "kali_devi", "parvathi_devi",
             "nataraja"],
}

# Spelling fixes so folder names match the project class list
EXTRA_ALIASES = {"anchitha": "anchita", "kartaka": "karkata"}


def seq_number(path):
    """Sequence number at the end of the file name: 'Kali devi 21' -> 21."""
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def build():
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw_dir"]
    label_alias = {**cfg.get("label_aliases", {}), **EXTRA_ALIASES}
    label_alias.pop("anchita", None)  # old config alias would conflict
    bani_alias = cfg.get("bani_aliases", {})
    rows = []

    for task in cfg["tasks"]:
        task_dir = raw / task
        if not task_dir.exists():
            continue
        for bani_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            bani = normalise(bani_dir.name, bani_alias)
            for label_dir in sorted(p for p in bani_dir.iterdir() if p.is_dir()):
                label = normalise(label_dir.name, label_alias)
                files = [f for f in label_dir.rglob("*")
                         if f.suffix.lower() in IMG_EXT]
                files.sort(key=lambda f: (seq_number(f) or 0, f.name))
                for i, f in enumerate(files):
                    seq = seq_number(f) or (i + 1)
                    is_ref = seq > TRAIN_MAX[task]
                    rows.append({
                        "path": f.relative_to(ROOT).as_posix(),
                        "task": task,
                        "label": label,
                        "bani": bani,
                        "level": "advanced",
                        "demonstrator_id": "ref" if is_ref else "learner",
                        "view": "front",
                        "seq": seq,
                        "split": "reference" if is_ref else "train",
                        "is_reference": int(is_ref),
                    })
    return cfg, rows


def main():
    cfg, rows = build()
    out = ROOT / cfg["paths"]["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "task", "label", "bani", "level", "demonstrator_id",
            "view", "seq", "split", "is_reference"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}\n")
    train = Counter((r["task"], r["label"]) for r in rows if not r["is_reference"])
    ref = Counter((r["task"], r["label"]) for r in rows if r["is_reference"])
    print(f"{'task':<11}{'label':<16}{'train':>6}{'ref':>5}")
    for key in sorted(set(train) | set(ref)):
        print(f"{key[0]:<11}{key[1]:<16}{train[key]:>6}{ref[key]:>5}")

    print("\nChecks:")
    problems = 0
    found = {(r["task"], r["label"]) for r in rows}
    for task, names in EXPECTED.items():
        for n in names:
            if (task, n) not in found:
                print(f"  MISSING folder: {task} / {n}")
                problems += 1
    for task, label in sorted(found):
        if label not in EXPECTED[task]:
            print(f"  UNKNOWN folder: {task} / {label}")
            problems += 1
    if problems == 0:
        print("  All classes found.")


if __name__ == "__main__":
    main()