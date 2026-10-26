from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalise(name, aliases=None):
    """Lowercase, replace spaces with _, and apply spelling aliases."""
    n = name.strip().lower().replace(" ", "_").replace("-", "_")
    if aliases:
        n = aliases.get(n.replace("_", ""), aliases.get(n, n))
    return n