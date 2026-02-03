from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .config import EvalConfig
from .evaluator import evaluate_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate The Well dataset and emit leaderboard JSON.")
    p.add_argument("--dataset", required=True, help="Well dataset name (e.g., helmholtz_staircase).")
    p.add_argument("--split", default="test", help="Split name (default: test).")
    p.add_argument("--model", default=None, help="HF model id. Default: polymathic-ai/FNO-{dataset}")
    p.add_argument("--Ti", type=int, default=4, help="n_steps_input (default: 4).")
    p.add_argument("--T", type=int, default=30, help="rollout steps (default: 30).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")

    p.add_argument("--base-path", default=os.environ.get("WELL_BASE_PATH", "/data"),
                   help="Base path where datasets live (default: /data or WELL_BASE_PATH).")
    p.add_argument("--out", default="./results", help="Output directory for JSON (default: ./results).")
    p.add_argument("--json", default=None, help="Optional output JSON filename or path.")
    p.add_argument("--no-download", action="store_true", help="Disable HF download if missing.")

    p.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda).")
    p.add_argument("--non-deterministic", action="store_true", help="Disable deterministic flags.")
    p.add_argument("--mininterval", type=float, default=1.0, help="tqdm mininterval (default: 1.0).")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = EvalConfig(
        dataset_name=args.dataset,
        split_name=args.split,
        model_id=args.model,
        n_steps_input=args.Ti,
        rollout=args.T,
        seed=args.seed,
        base_path=args.base_path,
        out_dir=args.out,
        output_json=args.json,
        download_if_missing=(not args.no_download),
        device=args.device,
        deterministic=(not args.non_deterministic),
        mininterval_tqdm=args.mininterval,
    )

    results = evaluate_dataset(cfg)
    # Print compact JSON to stdout for piping
    print("\n--- JSON (stdout) ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
