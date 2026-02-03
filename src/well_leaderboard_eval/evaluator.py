from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from huggingface_hub import snapshot_download

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.models import FNO
from the_well.benchmark.metrics import VRMSE as VRMSE_metric

from .config import EvalConfig
from .io import maybe_write_json


def _set_determinism(seed: int, deterministic: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _pack_input(current_window: torch.Tensor, const_fields: Optional[torch.Tensor]) -> Tuple[torch.Tensor, str]:
    # current_window: [B, Ti, H, W, F] or [B, Ti, H, W, D, F]
    if current_window.ndim == 5:
        x = rearrange(current_window, "B Ti H W F -> B (Ti F) H W")
        if const_fields is not None:
            c = rearrange(const_fields, "B H W Fc -> B Fc H W")
            x = torch.cat([x, c], dim=1)
        return x, "2d"
    if current_window.ndim == 6:
        x = rearrange(current_window, "B Ti H W D F -> B (Ti F) H W D")
        if const_fields is not None:
            c = rearrange(const_fields, "B H W D Fc -> B Fc H W D")
            x = torch.cat([x, c], dim=1)
        return x, "3d"
    raise ValueError(f"Unexpected input_fields ndim={current_window.ndim}")


def _unpack_pred(pred: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "2d":
        return rearrange(pred, "B F H W -> B 1 H W F")
    return rearrange(pred, "B F H W D -> B 1 H W D F")


def _get_const_fields(sample: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    if "constant_fields" in sample and sample["constant_fields"] is not None:
        if sample["constant_fields"].numel() > 0:
            return sample["constant_fields"].unsqueeze(0).to(device)
    return None


def _err_to_numpy(err_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
    # err_tensor is expected to have per-field error; we follow your mean(dim=(0,1))
    per_field = err_tensor.mean(dim=(0, 1)).detach().cpu().numpy()
    mean_all = float(per_field.mean())
    return per_field, mean_all


def _safe_mean_list(sum_vec: Optional[np.ndarray], n: int) -> Optional[list]:
    if n <= 0 or sum_vec is None:
        return None
    return (sum_vec / float(n)).tolist()


def _ensure_dataset(cfg: EvalConfig) -> str:
    dataset_path = os.path.join(cfg.base_path, cfg.dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path

    if not cfg.download_if_missing:
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Set download_if_missing=True or provide base_path."
        )

    snapshot_download(
        repo_id=f"polymathic-ai/{cfg.dataset_name}",
        repo_type="dataset",
        local_dir=dataset_path,
    )
    return dataset_path


def evaluate_dataset(cfg: EvalConfig) -> Dict[str, Any]:
    _set_determinism(cfg.seed, cfg.deterministic)

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    dataset_path = _ensure_dataset(cfg)

    model_id = cfg.model_id or f"polymathic-ai/FNO-{cfg.dataset_name}"

    # Load model
    model = FNO.from_pretrained(model_id).to(device)
    model.eval()

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print("Device:", device_name)
    print("Dataset:", cfg.dataset_name, "| split:", cfg.split_name)
    print("Model:", model_id)
    print("n_steps_input:", cfg.n_steps_input, "| rollout:", cfg.rollout)
    print("Dataset path:", dataset_path)

    # ---------------------------
    # 1-step (windowed)
    # ---------------------------
    print("\n" + "=" * 60)
    print("1-STEP (WINDOWED)")
    print("=" * 60)

    dset_1 = WellDataset(
        well_base_path=cfg.base_path,
        well_dataset_name=cfg.dataset_name,
        well_split_name=cfg.split_name,
        n_steps_input=cfg.n_steps_input,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    meta = dset_1.metadata

    s0 = dset_1[0]
    print("input_fields shape:", tuple(s0["input_fields"].shape))
    print("output_fields shape:", tuple(s0["output_fields"].shape))
    has_const0 = (
        ("constant_fields" in s0)
        and (s0["constant_fields"] is not None)
        and (s0["constant_fields"].numel() > 0)
    )
    print("has_constant_fields (sample0):", has_const0)
    if has_const0:
        print("constant_fields shape (sample0):", tuple(s0["constant_fields"].shape))

    one_step_vals = []
    per_field_sum_1 = None
    n_1 = 0

    with torch.no_grad():
        for i in tqdm(range(len(dset_1)), desc="1-step windowed", mininterval=cfg.mininterval_tqdm):
            s = dset_1[i]
            x0 = s["input_fields"].unsqueeze(0).to(device)
            y = s["output_fields"].unsqueeze(0).to(device)
            c = _get_const_fields(s, device)

            xin, mode = _pack_input(x0, c)
            pred = _unpack_pred(model(xin), mode)

            err = VRMSE_metric.eval(pred, y, meta)
            per_field, mean_all = _err_to_numpy(err)

            one_step_vals.append(mean_all)
            if per_field_sum_1 is None:
                per_field_sum_1 = np.zeros_like(per_field, dtype=np.float64)
            per_field_sum_1 += per_field.astype(np.float64)
            n_1 += 1

    one_step_windowed_mean = float(np.mean(one_step_vals)) if n_1 > 0 else float("nan")
    one_step_windowed_per_field_mean = _safe_mean_list(per_field_sum_1, n_1)

    print("1_step_windowed_mean:", one_step_windowed_mean)
    print("n_1step_samples:", n_1)

    # ---------------------------
    # Rollout
    # ---------------------------
    print("\n" + "=" * 60)
    print("ROLLOUT (normalized feedback)")
    print("=" * 60)

    if cfg.rollout <= 0:
        raise ValueError("rollout must be > 0")

    dset_r = WellDataset(
        well_base_path=cfg.base_path,
        well_dataset_name=cfg.dataset_name,
        well_split_name=cfg.split_name,
        n_steps_input=cfg.n_steps_input,
        n_steps_output=cfg.rollout,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )

    sum_per_step = np.zeros(cfg.rollout, dtype=np.float64)
    per_field_sum_step1 = None
    per_field_sum_6_12 = None
    per_field_sum_13_30 = None
    n_traj = 0

    # Precompute slice bounds safely (in case rollout < 30)
    # human 6..12 => indices 5..11 inclusive => python [5:12]
    a_6_12 = (5, min(12, cfg.rollout))
    # human 13..30 => indices 12..29 inclusive => python [12:30]
    a_13_30 = (12, min(30, cfg.rollout))

    with torch.no_grad():
        for i in tqdm(range(len(dset_r)), desc="rollout", mininterval=cfg.mininterval_tqdm):
            s = dset_r[i]
            current = s["input_fields"].unsqueeze(0).to(device)
            y = s["output_fields"].unsqueeze(0).to(device)
            c = _get_const_fields(s, device)

            step_vals = np.zeros(cfg.rollout, dtype=np.float64)
            per_field_steps = []

            for t in range(cfg.rollout):
                xin, mode = _pack_input(current, c)
                pred = _unpack_pred(model(xin), mode)

                gt = y[:, t : t + 1, ...]
                err = VRMSE_metric.eval(pred, gt, meta)
                per_field, mean_all = _err_to_numpy(err)

                per_field_steps.append(per_field)
                step_vals[t] = mean_all

                # normalized feedback (no denorm/renorm)
                pred_next = pred[:, 0, ...]
                current = torch.cat([current[:, 1:, ...], pred_next.unsqueeze(1)], dim=1)

            per_field_steps = np.stack(per_field_steps, axis=0)  # (T, F)

            if per_field_sum_step1 is None:
                F = per_field_steps.shape[1]
                per_field_sum_step1 = np.zeros((F,), dtype=np.float64)
                per_field_sum_6_12 = np.zeros((F,), dtype=np.float64)
                per_field_sum_13_30 = np.zeros((F,), dtype=np.float64)

            per_field_sum_step1 += per_field_steps[0].astype(np.float64)

            # Only compute if the slice is non-empty
            if a_6_12[0] < a_6_12[1]:
                per_field_sum_6_12 += per_field_steps[a_6_12[0] : a_6_12[1]].mean(axis=0).astype(np.float64)
            if a_13_30[0] < a_13_30[1]:
                per_field_sum_13_30 += per_field_steps[a_13_30[0] : a_13_30[1]].mean(axis=0).astype(np.float64)

            sum_per_step += step_vals
            n_traj += 1

    mean_curve = (sum_per_step / max(n_traj, 1)).tolist()
    rollout_step1_mean = float(mean_curve[0])
    rollout_6_12_mean = float(np.mean(mean_curve[a_6_12[0] : a_6_12[1]])) if a_6_12[0] < a_6_12[1] else float("nan")
    rollout_13_30_mean = float(np.mean(mean_curve[a_13_30[0] : a_13_30[1]])) if a_13_30[0] < a_13_30[1] else float("nan")

    rollout_step1_per_field_mean = _safe_mean_list(per_field_sum_step1, n_traj)
    rollout_6_12_per_field_mean = _safe_mean_list(per_field_sum_6_12, n_traj)
    rollout_13_30_per_field_mean = _safe_mean_list(per_field_sum_13_30, n_traj)

    print("rollout_step1_mean:", rollout_step1_mean)
    print("rollout_6_12_mean:", rollout_6_12_mean, f"(slice {a_6_12[0]}:{a_6_12[1]})")
    print("rollout_13_30_mean:", rollout_13_30_mean, f"(slice {a_13_30[0]}:{a_13_30[1]})")
    print("n_rollout_trajectories:", n_traj)

    # ---------------------------
    # 1-step rollout-aligned
    # ---------------------------
    print("\n" + "=" * 60)
    print("1-STEP (ROLLOUT-ALIGNED)")
    print("=" * 60)

    aligned_vals = []
    per_field_sum_aligned = None
    n_aligned = 0

    with torch.no_grad():
        for i in tqdm(range(len(dset_r)), desc="1-step rollout-aligned", mininterval=cfg.mininterval_tqdm):
            s = dset_r[i]
            current = s["input_fields"].unsqueeze(0).to(device)
            y = s["output_fields"].unsqueeze(0).to(device)
            c = _get_const_fields(s, device)

            xin, mode = _pack_input(current, c)
            pred = _unpack_pred(model(xin), mode)

            gt0 = y[:, 0:1, ...]
            err = VRMSE_metric.eval(pred, gt0, meta)
            per_field, mean_all = _err_to_numpy(err)

            aligned_vals.append(mean_all)
            if per_field_sum_aligned is None:
                per_field_sum_aligned = np.zeros_like(per_field, dtype=np.float64)
            per_field_sum_aligned += per_field.astype(np.float64)
            n_aligned += 1

    one_step_rollout_aligned_mean = float(np.mean(aligned_vals)) if n_aligned > 0 else float("nan")
    one_step_rollout_aligned_per_field_mean = _safe_mean_list(per_field_sum_aligned, n_aligned)

    print("1_step_rollout_aligned_mean:", one_step_rollout_aligned_mean)
    print("n_rollout_trajectories:", n_aligned)
    print("(Should be close to rollout_step1_mean:", rollout_step1_mean, ")")

    # ---------------------------
    # Results dict
    # ---------------------------
    results: Dict[str, Any] = {
        "dataset": cfg.dataset_name,
        "split": cfg.split_name,
        "model_id": model_id,
        "the_well_version": cfg.the_well_version,
        "code_version": cfg.code_version,

        "n_steps_input": cfg.n_steps_input,
        "rollout_steps": cfg.rollout,

        "1_step_windowed_mean": one_step_windowed_mean,
        "1_step_windowed_per_field_mean": one_step_windowed_per_field_mean,
        "n_1step_samples": int(n_1),

        "1_step_rollout_aligned_mean": one_step_rollout_aligned_mean,
        "1_step_rollout_aligned_per_field_mean": one_step_rollout_aligned_per_field_mean,
        "n_rollout_trajectories": int(n_traj),

        "rollout_step1_mean": rollout_step1_mean,
        "rollout_step1_per_field_mean": rollout_step1_per_field_mean,
        "rollout_6_12_mean": rollout_6_12_mean,
        "rollout_6_12_per_field_mean": rollout_6_12_per_field_mean,
        "rollout_13_30_mean": rollout_13_30_mean,
        "rollout_13_30_per_field_mean": rollout_13_30_per_field_mean,
        "rollout_per_step_mean_curve": mean_curve,

        "note": "rollout uses normalized feedback (no denorm/renorm)",
        "seed": cfg.seed,
        "deterministic": bool(cfg.deterministic),

        "device": str(device),
        "device_name": device_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Save JSON
    out_path = maybe_write_json(
        results=results,
        out_dir=cfg.out_dir,
        output_json=cfg.output_json,
        dataset=cfg.dataset_name,
        seed=cfg.seed,
        Ti=cfg.n_steps_input,
        T=cfg.rollout,
    )
    print("\nSaved:", out_path)
    results["output_path"] = out_path

    return results
