from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_agent.config import ProjectConfig, load_config, save_config
from traffic_agent.data.dataset import TrafficWindowDataset
from traffic_agent.data.generate_synthetic import save_synthetic_dataset
from traffic_agent.data.loader import load_traffic_npz
from traffic_agent.data.prepare_real_dataset import build_correlation_adjacency
from traffic_agent.data.preprocessing import inverse_transform_feature, preprocess_data
from traffic_agent.models.model_factory import NON_TRAINABLE_MODELS, SUPPORTED_MODELS, build_model, count_parameters
from traffic_agent.training.metrics import mae, mape, masked_mae, masked_mape, masked_rmse, rmse


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible demo runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dataset_available(path: str, seed: int) -> None:
    """Generate only the synthetic quick-start dataset; never fake missing real data."""
    data_path = Path(path)
    if data_path.exists():
        return
    if data_path.as_posix().endswith("data/sample/synthetic_traffic.npz"):
        save_synthetic_dataset(data_path, seed=seed)
        return
    raise FileNotFoundError(
        f"Dataset not found: {data_path}. For real data, run "
        "`python -m traffic_agent.data.prepare_real_dataset ...` first. "
        "The training script will not substitute synthetic data for a missing real dataset."
    )


def make_run_dir(config: ProjectConfig, model_name: str, run_suffix: str | None = None) -> Path:
    suffix = f"_{run_suffix}" if run_suffix else ""
    base = Path(config.outputs.dir) / f"{config.outputs.run_name}_{model_name}{suffix}"
    if config.outputs.allow_overwrite or not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.outputs.dir) / f"{config.outputs.run_name}_{model_name}{suffix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Compute average loss on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += float(loss.item()) * x.shape[0]
            total_items += x.shape[0]
    return total_loss / max(total_items, 1)


def make_loss(name: str) -> nn.Module:
    """Create a training loss by name."""
    if name in {"mae", "masked_mae"}:
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError("training.loss must be one of: mae, masked_mae, mse, huber")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def gpu_info(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"gpu_name": None, "cuda_version": torch.version.cuda, "gpu_total_memory_mb": None}
    index = device.index or 0
    props = torch.cuda.get_device_properties(index)
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "cuda_version": torch.version.cuda,
        "gpu_total_memory_mb": round(props.total_memory / 1024**2, 2),
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
    target_feature_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect test predictions, labels, and recent history means in original units."""
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    history_means: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y_np = y.numpy()
            history_speed = x.cpu().numpy()[..., target_feature_index].mean(axis=1)
            preds.append(inverse_transform_feature(pred, scaler, target_feature_index))
            trues.append(inverse_transform_feature(y_np, scaler, target_feature_index))
            history_means.append(inverse_transform_feature(history_speed, scaler, target_feature_index))
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(history_means)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _range_from_timestamps(timestamps: list[str], start: int, end: int) -> dict[str, str | int | None]:
    if not timestamps or start >= len(timestamps):
        return {"start": None, "end": None, "start_index": start, "end_index": max(start, end - 1)}
    end_index = min(max(start, end - 1), len(timestamps) - 1)
    return {
        "start": timestamps[start],
        "end": timestamps[end_index],
        "start_index": start,
        "end_index": end_index,
    }


def _write_run_summary(run_dir: Path, metrics: dict[str, object]) -> None:
    synthetic_note = (
        "This is a synthetic quick-start run and should not be used as evidence of real traffic performance."
        if metrics["is_synthetic_data"]
        else "This run is marked as real-data based. Verify dataset provenance before using it in a resume."
    )
    lines = [
        "# Run Summary",
        "",
        f"- Dataset: {metrics['dataset_name']}",
        f"- Synthetic: {metrics['is_synthetic_data']}",
        f"- Model: {metrics['model_name']}",
        f"- Horizon: {metrics['horizon']}",
        f"- Adjacency type: {metrics['adjacency_type']}",
        f"- MAE: {metrics['mae']:.6f}",
        f"- RMSE: {metrics['rmse']:.6f}",
        f"- MAPE: {metrics['mape']:.6f}",
        f"- Masked MAE: {metrics['masked_mae']:.6f}",
        f"- Masked RMSE: {metrics['masked_rmse']:.6f}",
        f"- Masked MAPE: {metrics['masked_mape']:.6f}",
        f"- Best epoch: {metrics.get('best_epoch')}",
        f"- Best val loss: {metrics.get('best_val_loss')}",
        "",
        "## Resume Use",
        "",
        synthetic_note,
        "",
        "## Limitations",
        "",
        "- Metrics are computed only for this local split and configuration.",
        "- MAPE uses a 1 speed-unit denominator floor; masked MAPE ignores near-zero true speeds.",
        "- Full models are paper-inspired implementations, not official reproduction results.",
        "- Congestion risk and explanations are offline analysis aids, not traffic control decisions.",
    ]
    (run_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _apply_adjacency_mode(
    adjacency: np.ndarray,
    data: np.ndarray,
    train_ratio: float,
    adjacency_mode: str | None,
) -> tuple[np.ndarray, str]:
    if adjacency_mode in {None, "physical_graph", "physical"}:
        return adjacency, "physical"
    if adjacency_mode in {"no_graph", "identity", "temporal_only"}:
        return np.eye(adjacency.shape[0], dtype=np.float32), "none"
    if adjacency_mode == "correlation_graph":
        return build_correlation_adjacency(data, train_ratio=train_ratio), "correlation"
    if adjacency_mode == "adaptive_graph":
        return adjacency, "adaptive"
    raise ValueError(f"Unknown adjacency mode: {adjacency_mode}")


def train_model(
    config_path: str,
    model_name: str,
    horizon: int | None = None,
    seed: int | None = None,
    run_suffix: str | None = None,
    adjacency_mode: str | None = None,
) -> Path:
    """Train or evaluate a forecasting model and save run artifacts."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(SUPPORTED_MODELS)}.")

    config = load_config(config_path)
    if horizon is not None:
        config.data.horizon = horizon
    if seed is not None:
        config.seed = seed
    set_seed(config.seed)
    ensure_dataset_available(config.data.path, config.seed)

    data, adjacency, metadata = load_traffic_npz(config.data.path)
    adjacency, computed_adjacency_type = _apply_adjacency_mode(
        adjacency,
        data,
        config.data.train_ratio,
        adjacency_mode,
    )
    if adjacency_mode is None:
        computed_adjacency_type = str(metadata.get("adjacency_type", computed_adjacency_type))

    feature_names = metadata["feature_names"]
    if config.data.target_feature not in feature_names:
        raise ValueError(f"target_feature={config.data.target_feature!r} not found in feature_names={feature_names}.")
    target_index = feature_names.index(config.data.target_feature)
    processed = preprocess_data(data, config.data.train_ratio, config.data.val_ratio)

    train_dataset = TrafficWindowDataset(processed.train, config.data.input_steps, config.data.horizon, target_index)
    val_dataset = TrafficWindowDataset(processed.val, config.data.input_steps, config.data.horizon, target_index)
    test_dataset = TrafficWindowDataset(processed.test, config.data.input_steps, config.data.horizon, target_index)
    worker_kwargs = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }
    if config.training.num_workers > 0:
        worker_kwargs["persistent_workers"] = config.training.persistent_workers
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, **worker_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, **worker_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, **worker_kwargs)

    device = resolve_device(config.training.device)
    info = gpu_info(device)
    if device.type == "cuda":
        print(
            "Using GPU: "
            f"{info['gpu_name']} total_memory_mb={info['gpu_total_memory_mb']} cuda={info['cuda_version']}"
        )
    model = build_model(
        model_name,
        config,
        num_nodes=data.shape[1],
        num_features=data.shape[2],
        adjacency=torch.from_numpy(adjacency).to(device),
        target_feature_index=target_index,
    ).to(device)
    if config.training.compile_model and model_name not in NON_TRAINABLE_MODELS:
        try:
            model = torch.compile(model)
        except Exception as exc:  # pragma: no cover - compile support is environment-specific
            print(f"WARNING: torch.compile failed, falling back to eager mode: {exc}")
            config.training.compile_model = False

    run_dir = make_run_dir(config, model_name, run_suffix=run_suffix)
    criterion = make_loss(config.training.loss)
    history: list[dict[str, float]] = []
    train_start = time.perf_counter()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.use_amp and device.type == "cuda")
    if model_name not in NON_TRAINABLE_MODELS:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        for epoch in range(1, config.training.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_items = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=config.training.use_amp and device.type == "cuda",
                ):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    failure = [
                        "# Training Failure",
                        "",
                        f"- epoch: {epoch}",
                        f"- loss: {float(loss.detach().cpu())}",
                        "- Possible causes: too high learning rate, unstable normalization, bad input scale, AMP overflow.",
                    ]
                    (run_dir / "failure_report.md").write_text("\n".join(failure), encoding="utf-8")
                    raise FloatingPointError(f"Training loss became NaN/Inf. See {run_dir / 'failure_report.md'}")
                scaler.scale(loss).backward()
                if config.training.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * x.shape[0]
                total_items += x.shape[0]
            train_loss = total_loss / max(total_items, 1)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            epoch_seconds = time.perf_counter() - epoch_start
            gpu_allocated = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            gpu_reserved = torch.cuda.memory_reserved(device) / 1024**2 if device.type == "cuda" else 0.0
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": epoch_seconds,
                    "gpu_memory_allocated_mb": gpu_allocated,
                    "gpu_memory_reserved_mb": gpu_reserved,
                }
            )
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
            if config.training.patience is not None and epochs_without_improvement >= config.training.patience:
                early_stopped = True
                print(f"early stopping at epoch={epoch}")
                break
    else:
        train_loss = evaluate_loss(model, train_loader, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        best_val_loss = val_loss
        history.append(
            {
                "epoch": 0,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": 0.0,
                "epoch_seconds": 0.0,
                "gpu_memory_allocated_mb": 0.0,
                "gpu_memory_reserved_mb": 0.0,
            }
        )
        if model_name == "seasonal_naive":
            print("seasonal_naive may fall back to historical average when input window is shorter than one day.")
        print(f"baseline train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    best_model_path = run_dir / "best_model.pt"
    if model_name not in NON_TRAINABLE_MODELS and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    y_pred, y_true, history_mean = collect_predictions(model, test_loader, device, processed.scaler, target_index)
    save_config(config, run_dir / "config.yaml")
    import pandas as pd

    pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)

    timestamps = list(metadata.get("timestamps", []))
    train_end, val_end = processed.split_indices
    test_timestamps = timestamps[val_end + config.data.input_steps : val_end + config.data.input_steps + len(y_true)]
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        history_mean=history_mean.astype(np.float32),
        node_ids=np.array(metadata["node_ids"]),
        timestamps=np.array(test_timestamps),
        horizon=np.array(config.data.horizon),
        is_synthetic_data=np.array(metadata["is_synthetic_data"]),
    )

    metrics = {
        "dataset_name": metadata.get("dataset_name", Path(config.data.path).stem),
        "dataset_path": metadata["path"],
        "is_synthetic_data": bool(metadata["is_synthetic_data"]),
        "adjacency_type": computed_adjacency_type,
        "num_nodes": int(data.shape[1]),
        "num_timesteps": int(data.shape[0]),
        "num_features": int(data.shape[2]),
        "train_time_range": _range_from_timestamps(timestamps, 0, train_end),
        "val_time_range": _range_from_timestamps(timestamps, train_end, val_end),
        "test_time_range": _range_from_timestamps(timestamps, val_end, data.shape[0]),
        "model_name": model_name,
        "parameter_count": count_parameters(model),
        "input_steps": config.data.input_steps,
        "horizon": config.data.horizon,
        "seed": config.seed,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "masked_mae": masked_mae(y_true, y_pred),
        "masked_rmse": masked_rmse(y_true, y_pred),
        "masked_mape": masked_mape(y_true, y_pred),
        "mape_denominator_floor": 1.0,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_seconds": time.perf_counter() - train_start,
        "avg_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])) if history else 0.0,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "git_commit": _git_commit(),
        "device": str(device),
        "gpu_name": info["gpu_name"],
        "use_amp": bool(config.training.use_amp and device.type == "cuda"),
        "compile_model": bool(config.training.compile_model),
        "batch_size": config.training.batch_size,
        "epochs_configured": config.training.epochs,
        "epochs_trained": len([row for row in history if row["epoch"] > 0]),
        "early_stopped": early_stopped,
        "loss_name": config.training.loss,
        "history": history,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    _write_run_summary(run_dir, metrics)

    if model_name not in NON_TRAINABLE_MODELS:
        torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"Saved run artifacts to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a traffic forecasting model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--adjacency-mode", default=None)
    args = parser.parse_args()
    train_model(args.config, args.model, args.horizon, args.seed, args.run_suffix, args.adjacency_mode)


if __name__ == "__main__":
    main()
