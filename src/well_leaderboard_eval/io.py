from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_output_name(dataset: str, seed: int, Ti: int, T: int) -> str:
    return f"{dataset}_results_seed{seed}_Ti{Ti}_T{T}.json"


def write_json(results: Dict[str, Any], out_dir: str, filename: str) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path


def maybe_write_json(results: Dict[str, Any], out_dir: str, output_json: Optional[str],
                     dataset: str, seed: int, Ti: int, T: int) -> str:
    if output_json is not None:
        # If user gave a file name (not a path), put it under out_dir
        if os.path.dirname(output_json) == "":
            filename = output_json
            return write_json(results, out_dir, filename)
        # Else treat as full/relative path
        ensure_dir(os.path.dirname(output_json))
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        return output_json

    filename = default_output_name(dataset, seed, Ti, T)
    return write_json(results, out_dir, filename)
