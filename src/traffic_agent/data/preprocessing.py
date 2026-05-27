from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class PreprocessingResult:
    """Container for time-ordered splits and fitted scaler."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scaler: StandardScaler
    train_mean: np.ndarray
    split_indices: tuple[int, int]


def split_time_ordered(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Split data by time without shuffling."""
    if data.ndim != 3:
        raise ValueError(f"Expected data shape [T, N, F], got {data.shape}.")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be valid fractions.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    total_steps = data.shape[0]
    train_end = int(total_steps * train_ratio)
    val_end = train_end + int(total_steps * val_ratio)
    if train_end == 0 or val_end <= train_end or val_end >= total_steps:
        raise ValueError("Split ratios create an empty train, val, or test split.")
    return data[:train_end], data[train_end:val_end], data[val_end:], (train_end, val_end)


def _fill_by_time(values: np.ndarray, fallback_feature_mean: np.ndarray | None = None) -> np.ndarray:
    original_shape = values.shape
    flat = values.reshape(values.shape[0], -1)
    filled = pd.DataFrame(flat).ffill().bfill().to_numpy(dtype=np.float32)
    if fallback_feature_mean is not None and np.isnan(filled).any():
        fallback = np.tile(fallback_feature_mean, values.shape[1])
        nan_rows, nan_cols = np.where(np.isnan(filled))
        filled[nan_rows, nan_cols] = fallback[nan_cols]
    return filled.reshape(original_shape).astype(np.float32)


def preprocess_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> PreprocessingResult:
    """Fill missing values and standardize features using train split statistics only."""
    train_raw, val_raw, test_raw, split_indices = split_time_ordered(data, train_ratio, val_ratio)
    train_filled_initial = _fill_by_time(train_raw)
    train_mean = np.nanmean(train_filled_initial.reshape(-1, train_raw.shape[-1]), axis=0)
    train_mean = np.where(np.isnan(train_mean), 0.0, train_mean).astype(np.float32)

    train_filled = _fill_by_time(train_raw, train_mean)
    val_filled = _fill_by_time(val_raw, train_mean)
    test_filled = _fill_by_time(test_raw, train_mean)

    scaler = StandardScaler()
    scaler.fit(train_filled.reshape(-1, train_filled.shape[-1]))

    train = _transform_split(train_filled, scaler)
    val = _transform_split(val_filled, scaler)
    test = _transform_split(test_filled, scaler)
    return PreprocessingResult(train, val, test, scaler, train_mean, split_indices)


def _transform_split(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = values.shape
    transformed = scaler.transform(values.reshape(-1, shape[-1]))
    return transformed.reshape(shape).astype(np.float32)


def inverse_transform_feature(
    values: np.ndarray,
    scaler: StandardScaler,
    feature_index: int = 0,
) -> np.ndarray:
    """Inverse transform one feature, useful for restoring speed predictions."""
    return values * float(scaler.scale_[feature_index]) + float(scaler.mean_[feature_index])
