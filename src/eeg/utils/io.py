import json
from typing import Any

import yaml


def load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf8") as fh:
        return yaml.safe_load(fh)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)
