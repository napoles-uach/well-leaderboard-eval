from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    # Core
    dataset_name: str = "helmholtz_staircase"
    split_name: str = "test"
    model_id: Optional[str] = None
    n_steps_input: int = 4
    rollout: int = 30
    seed: int = 0

    # Paths / IO
    base_path: str = "/data"        # where datasets are stored (HF snapshot_download writes here)
    out_dir: str = "./results"
    output_json: Optional[str] = None
    download_if_missing: bool = True

    # Runtime
    device: str = "cuda"            # "cuda" or "cpu"
    deterministic: bool = True
    mininterval_tqdm: float = 1.0

    # Metadata / tracing
    the_well_version: str = "v1.1.0"
    code_version: str = "0.1.0"
