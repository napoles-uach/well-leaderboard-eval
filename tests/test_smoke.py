import os
import pytest

from well_leaderboard_eval import EvalConfig, evaluate_dataset


@pytest.mark.skipif(os.environ.get("RUN_WELL_SMOKE", "0") != "1", reason="set RUN_WELL_SMOKE=1 to run")
def test_smoke_runs_and_has_keys(tmp_path):
    cfg = EvalConfig(
        dataset_name=os.environ.get("WELL_DATASET", "helmholtz_staircase"),
        split_name="test",
        n_steps_input=4,
        rollout=5,
        seed=0,
        base_path=os.environ.get("WELL_BASE_PATH", "/data"),
        out_dir=str(tmp_path),
        download_if_missing=True,
        device="cuda",
        deterministic=True,
    )

    res = evaluate_dataset(cfg)
    for k in [
        "dataset",
        "split",
        "model_id",
        "1_step_windowed_mean",
        "rollout_step1_mean",
        "rollout_per_step_mean_curve",
        "output_path",
    ]:
        assert k in res
