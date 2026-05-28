from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_agent.training.metrics import mae, rmse


def load_predictions(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"predictions.npz not found: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def error_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute MAE/RMSE for each forecast horizon step."""
    rows = []
    for step in range(y_true.shape[1]):
        rows.append(
            {
                "horizon_step": step + 1,
                "mae": mae(y_true[:, step], y_pred[:, step]),
                "rmse": rmse(y_true[:, step], y_pred[:, step]),
            }
        )
    return pd.DataFrame(rows)


def error_by_node(y_true: np.ndarray, y_pred: np.ndarray, node_ids: list[str] | None = None) -> pd.DataFrame:
    """Compute node-wise errors."""
    ids = node_ids or [f"node_{idx}" for idx in range(y_true.shape[-1])]
    rows = []
    for idx, node_id in enumerate(ids):
        rows.append(
            {
                "node_id": node_id,
                "mae": mae(y_true[:, :, idx], y_pred[:, :, idx]),
                "rmse": rmse(y_true[:, :, idx], y_pred[:, :, idx]),
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def error_by_time_of_day(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: list[str] | None = None,
    steps_per_day: int = 288,
) -> pd.DataFrame:
    """Approximate time-of-day error buckets from timestamps or sample index."""
    sample_error = np.abs(y_true - y_pred).mean(axis=(1, 2))
    if timestamps:
        parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        hour = parsed.dt.hour.fillna(np.arange(len(sample_error)) % steps_per_day // 12).astype(int)
    else:
        hour = pd.Series((np.arange(len(sample_error)) % steps_per_day) // 12)
    return pd.DataFrame({"hour": hour[: len(sample_error)], "mae": sample_error}).groupby("hour", as_index=False).mean()


def residual_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Summarize residual distribution."""
    residual = (y_pred - y_true).reshape(-1)
    return {
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "p05": float(np.percentile(residual, 5)),
        "p50": float(np.percentile(residual, 50)),
        "p95": float(np.percentile(residual, 95)),
    }


def worst_case_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_ids: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return sample/horizon/node cells with the largest absolute errors."""
    ids = node_ids or [f"node_{idx}" for idx in range(y_true.shape[-1])]
    errors = np.abs(y_true - y_pred)
    flat_indices = np.argsort(errors.reshape(-1))[::-1][:top_k]
    rows = []
    for flat in flat_indices:
        sample, horizon, node = np.unravel_index(flat, errors.shape)
        rows.append(
            {
                "sample_index": int(sample),
                "horizon_step": int(horizon + 1),
                "node_id": ids[int(node)],
                "absolute_error": float(errors[sample, horizon, node]),
                "y_true": float(y_true[sample, horizon, node]),
                "y_pred": float(y_pred[sample, horizon, node]),
            }
        )
    return rows


def compare_models_by_horizon(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Compare multiple run directories by horizon step."""
    frames = []
    for run_dir in run_dirs:
        payload = load_predictions(run_dir)
        frame = error_by_horizon(payload["y_true"], payload["y_pred"])
        frame["run_name"] = Path(run_dir).name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def error_by_speed_bin(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Group errors by true-speed bins."""
    bins = [0, 20, 35, 50, np.inf]
    labels = ["0-20", "20-35", "35-50", "50+"]
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    groups = pd.cut(flat_true, bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        mask = np.asarray(groups == label)
        if mask.any():
            rows.append(
                {
                    "speed_bin": label,
                    "sample_count": int(mask.sum()),
                    "mae": mae(flat_true[mask], flat_pred[mask]),
                    "rmse": rmse(flat_true[mask], flat_pred[mask]),
                }
            )
    return pd.DataFrame(rows)


def error_by_volatility(y_true: np.ndarray, y_pred: np.ndarray, history_mean: np.ndarray | None = None) -> pd.DataFrame:
    """Group forecast errors by future-window volatility."""
    volatility = np.std(y_true, axis=1).mean(axis=1)
    buckets = pd.qcut(volatility, q=min(4, len(np.unique(volatility))), duplicates="drop")
    sample_error = np.abs(y_true - y_pred).mean(axis=(1, 2))
    frame = pd.DataFrame({"volatility_bin": buckets.astype(str), "mae": sample_error})
    return frame.groupby("volatility_bin", as_index=False).agg(sample_count=("mae", "size"), mae=("mae", "mean"))


def compare_model_failures(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    top_k: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Find samples where one model is much better than another."""
    base = load_predictions(baseline_run_dir)
    cand = load_predictions(candidate_run_dir)
    true = base["y_true"]
    base_err = np.abs(true - base["y_pred"]).mean(axis=(1, 2))
    cand_err = np.abs(true - cand["y_pred"]).mean(axis=(1, 2))
    delta = base_err - cand_err
    candidate_wins = np.argsort(-delta)[:top_k]
    baseline_wins = np.argsort(delta)[:top_k]
    return {
        "baseline_failed_candidate_succeeded": [
            {"sample_index": int(idx), "baseline_mae": float(base_err[idx]), "candidate_mae": float(cand_err[idx])}
            for idx in candidate_wins
            if delta[idx] > 0
        ],
        "candidate_failed_baseline_succeeded": [
            {"sample_index": int(idx), "baseline_mae": float(base_err[idx]), "candidate_mae": float(cand_err[idx])}
            for idx in baseline_wins
            if delta[idx] < 0
        ],
    }


def explain_failure_case(
    run_dir: str | Path,
    sample_index: int,
    node_index: int,
) -> str:
    """Generate a markdown explanation for one failure case."""
    payload = load_predictions(run_dir)
    node_ids = [str(item) for item in payload["node_ids"].tolist()]
    true_values = payload["y_true"][sample_index, :, node_index].round(3).tolist()
    pred_values = payload["y_pred"][sample_index, :, node_index].round(3).tolist()
    history = (
        payload["history_mean"][sample_index, node_index].round(3).item()
        if "history_mean" in payload
        else "unknown"
    )
    return "\n".join(
        [
            "# Failure Case Explanation",
            "",
            f"- run: {Path(run_dir).name}",
            f"- node_id: {node_ids[node_index]}",
            f"- sample_index: {sample_index}",
            f"- recent_history_mean: {history}",
            f"- true_future: {true_values}",
            f"- predicted_future: {pred_values}",
            "",
            "Possible reason: the future segment may contain abrupt changes or low-speed readings that are hard "
            "for the current model and features to predict.",
            "",
            "Limitation: this is a heuristic diagnostic, not a causal explanation.",
        ]
    )


def worst_case_report(run_dir: str | Path, output: str | Path, top_k: int = 20) -> Path:
    """Write a markdown report of worst forecast cells."""
    payload = load_predictions(run_dir)
    node_ids = [str(item) for item in payload["node_ids"].tolist()]
    cases = worst_case_segments(payload["y_true"], payload["y_pred"], node_ids, top_k=top_k)
    lines = ["# Worst Case Report", ""]
    for case in cases:
        lines.append(
            f"- sample={case['sample_index']} horizon={case['horizon_step']} node={case['node_id']} "
            f"abs_error={case['absolute_error']:.4f}"
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
