from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from src.index_slicing import get_cols_idx, get_slice

VALID_MODES = {"lags", "seasonal_lags", "calendar", "fourier"}


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids: np.ndarray,
    targets_ids: np.ndarray,
    id_column: Union[str, Sequence[str]] = "id",
    date_column: Union[str, Sequence[str]] = "datetime",
    target_column: str = "target",
    feature_modes: List[str] = None,
    n_lags: int = 8,
    seasonal_period: int = 52,
    n_seasonal_lags: int = 3,
    fourier_order: int = 3,
):
    """
    Поддерживаемые режимы:
        'lags'
        'seasonal_lags'
        'calendar'      
        'fourier' 
    """
    unknown = set(feature_modes) - VALID_MODES
    if unknown:
        raise ValueError(f"неизвестные режимы: {unknown}")

    history_size = features_ids.shape[1]

    feature_blocks = []
    categorical_col_indices = []
    col_cursor = 0

    id_feat = get_slice(df, (targets_ids[:, :1], get_cols_idx(df, id_column)))
    id_feat = id_feat.astype(object)
    id_feat[:, 0] = id_feat[:, 0].astype(str)
    
    feature_blocks.append(id_feat)
    categorical_col_indices.extend(range(col_cursor, col_cursor + id_feat.shape[1]))
    col_cursor += id_feat.shape[1]

    if "lags" in feature_modes:
        if n_lags > history_size:
            raise ValueError(
                f"длина лагов больше истории"
            )
        lag_idx = features_ids[:, -n_lags:]
        lags = get_slice(df, (lag_idx, get_cols_idx(df, target_column)))
        feature_blocks.append(lags.astype(object))
        col_cursor += lags.shape[1]

    if "seasonal_lags" in feature_modes:

        min_history = seasonal_period + n_seasonal_lags - 1

        if history_size < min_history:
            raise ValueError(
                "слишком маленький горизонт истории"
            )
        
        s_cols = [history_size - seasonal_period - i for i in range(n_seasonal_lags)]
        s_lags = get_slice(df, (features_ids[:, s_cols], get_cols_idx(df, target_column)))
        feature_blocks.append(s_lags.astype(object))
        col_cursor += s_lags.shape[1]

    if "calendar" in feature_modes:
        target_dates = df.iloc[targets_ids[:, 0]][date_column]

        week    = target_dates.dt.isocalendar().week.astype(int).values
        month   = target_dates.dt.month.values
        quarter = target_dates.dt.quarter.values

        cal_feat = np.column_stack([week, month, quarter]).astype(object)

        for j in range(cal_feat.shape[1]):
            cal_feat[:, j] = cal_feat[:, j].astype(str)

        feature_blocks.append(cal_feat)
        categorical_col_indices.extend(range(col_cursor, col_cursor + cal_feat.shape[1]))
        col_cursor += cal_feat.shape[1]

    if "fourier" in feature_modes:
        target_dates = df.iloc[targets_ids[:, 0]][date_column]
        t = target_dates.dt.isocalendar().week.astype(int).values.astype(float)
        fourier_parts = []
        for k in range(1, fourier_order + 1):
            fourier_parts.append(np.sin(2 * np.pi * k * t / seasonal_period))
            fourier_parts.append(np.cos(2 * np.pi * k * t / seasonal_period))

        fourier_feat = np.column_stack(fourier_parts)
        feature_blocks.append(fourier_feat.astype(object))

        col_cursor += fourier_feat.shape[1]

    features_obj = np.hstack(feature_blocks)
    categorical_features_idx = np.array(categorical_col_indices)

    targets = get_slice(df, (targets_ids, get_cols_idx(df, target_column)))
    return features_obj, targets, categorical_features_idx
