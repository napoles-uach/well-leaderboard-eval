import os
import sys

# add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from well_leaderboard_eval import EvalConfig, evaluate_dataset

cfg = EvalConfig(
    dataset_name="helmholtz_staircase",
    split_name="test",
    n_steps_input=4,
    rollout=30,
    seed=0,
    base_path="/data",
    out_dir="./results",
    download_if_missing=True,
    device="cuda",
    deterministic=True,
)

res = evaluate_dataset(cfg)
print(res["output_path"])
