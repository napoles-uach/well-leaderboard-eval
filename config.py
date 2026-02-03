# src/well_leaderboard_eval/config.py
from dataclasses import dataclass

@dataclass
class EvalConfig:
    dataset_name: str = "helmholtz_staircase"
    split_name: str = "test"
    model_id: str | None = None
    n_steps_input: int = 4
    rollout: int = 30
    seed: int = 0

    # IO
    base_path: str = "/data"          # o local
    out_dir: str = "./results"
    download_if_missing: bool = True

    # Determinism
    deterministic: bool = True
