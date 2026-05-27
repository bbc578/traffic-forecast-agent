from __future__ import annotations

import argparse
import json
import random
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
from traffic_agent.data.preprocessing import inverse_transform_feature, preprocess_data
from traffic_agent.models.baseline import HistoricalAverageBaseline
from traffic_agent.models.lstm import LSTMForecaster
from traffic_agent.models.stgcn_lite import STGCNLite
from traffic_agent.training.metrics import mae, mape, rmse

SUPPORTED_MODELS = {"historical_average", "lstm", "stgcn_lite"}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible demo runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_demo_data(path: str, seed: int) -> None:
    data_path = Path(path)
    if data_path.exists():
        return
    if data_path.as_posix().endswith("data/sample/synthetic_traffic.npz"):
        save_synthetic_dataset(data_path, seed=seed)
        return
    raise FileNotFoundError(
        f"Dataset not found: {data_path}. Generate synthetic demo data or provide a local .npz file."
    )


def make_run_dir(config: ProjectConfig, model_name: str) -> Path:
    base = Path(config.outputs.dir) / f"{config.outputs.run_name}_{model_name}"
    if config.outputs.allow_overwrite or not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.outputs.dir) / f"{config.outputs.run_name}_{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_model(
    model_name: str,
    config: ProjectConfig,
    num_nodes: int,
    num_features: int,
    adjacency: np.ndarray,
    target_feature_index: int,
    device: torch.device,
) -> nn.Module:
    """Construct a supported model by name."""
    horizon = config.data.horizon
    if model_name == "historical_average":
        return HistoricalAverageBaseline(horizon, target_feature_index).to(device)
    if model_name == "lstm":
        return LSTMForecaster(
            num_nodes=num_nodes,
            num_features=num_features,
            horizon=horizon,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        ).to(device)
    if model_name == "stgcn_lite":
        return STGCNLite(
            num_nodes=num_nodes,
            num_features=num_features,
            horizon=horizon,
            adjacency=torch.from_numpy(adjacency).to(device),
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        ).to(device)
    raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(SUPPORTED_MODELS)}.")


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
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    return total_loss / max(total_items, 1)


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
            x_np = x.cpu().numpy()
            history_speed = x_np[..., target_feature_index].mean(axis=1)
            preds.append(inverse_transform_feature(pred, scaler, target_feature_index))
            trues.append(inverse_transform_feature(y_np, scaler, target_feature_index))
            history_means.append(
                inverse_transform_feature(history_speed, scaler, target_feature_index)
            )
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(history_means)


def train_model(config_path: str, model_name: str) -> Path:
    """Train or evaluate a forecasting model and save run artifacts."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(SUPPORTED_MODELS)}.")

    config = load_config(config_path)
    set_seed(config.seed)
    ensure_demo_data(config.data.path, config.seed)

    data, adjacency, metadata = load_traffic_npz(config.data.path)
    feature_names = metadata["feature_names"]
    if config.data.target_feature not in feature_names:
        raise ValueError(
            f"target_feature={config.data.target_feature!r} not found in feature_names={feature_names}."
        )
    target_index = feature_names.index(config.data.target_feature)
    processed = preprocess_data(data, config.data.train_ratio, config.data.val_ratio)

    train_dataset = TrafficWindowDataset(
        processed.train, config.data.input_steps, config.data.horizon, target_index
    )
    val_dataset = TrafficWindowDataset(
        processed.val, config.data.input_steps, config.data.horizon, target_index
    )
    test_dataset = TrafficWindowDataset(
        processed.test, config.data.input_steps, config.data.horizon, target_index
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name,
        config,
        num_nodes=data.shape[1],
        num_features=data.shape[2],
        adjacency=adjacency,
        target_feature_index=target_index,
        device=device,
    )

    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []
    if model_name != "historical_average":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        for epoch in range(1, config.training.epochs + 1):
            model.train()
            total_loss = 0.0
            total_items = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * x.shape[0]
                total_items += x.shape[0]
            train_loss = total_loss / max(total_items, 1)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    else:
        train_loss = evaluate_loss(model, train_loader, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": 0, "train_loss": train_loss, "val_loss": val_loss})
        print(f"baseline train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    y_pred, y_true, history_mean = collect_predictions(
        model, test_loader, device, processed.scaler, target_index
    )
    run_dir = make_run_dir(config, model_name)
    save_config(config, run_dir / "config.yaml")
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        history_mean=history_mean.astype(np.float32),
        node_ids=np.array(metadata["node_ids"]),
        horizon=np.array(config.data.horizon),
        is_synthetic_data=np.array(metadata["is_synthetic_data"]),
    )

    metrics = {
        "model_name": model_name,
        "dataset_path": metadata["path"],
        "input_steps": config.data.input_steps,
        "horizon": config.data.horizon,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "is_synthetic_data": bool(metadata["is_synthetic_data"]),
        "device": str(device),
        "history": history,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if model_name != "historical_average":
        torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"Saved run artifacts to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a traffic forecasting model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    args = parser.parse_args()
    train_model(args.config, args.model)


if __name__ == "__main__":
    main()
