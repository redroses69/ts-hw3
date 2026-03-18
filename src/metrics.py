from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return 200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def get_smape_table(all_results: dict, sample_ids: List[str]) -> pd.DataFrame:

    rows = []
    for name, res in all_results.items():
        row = {
            "model": name,
            "SMAPE_overall": smape(res["true_value"].values, res["predicted_value"].values),
        }

        for sid in sample_ids:
            g = res[res["sensor_id"] == sid]
            row[sid] = smape(g["true_value"].values, g["predicted_value"].values)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("SMAPE_overall")
        .round(2)
    )


def get_stl_seasonality_strength(values: np.ndarray, period: int = 52) -> float:
    stl = STL(values, period=period, robust=True).fit()
    var_remainder = np.var(stl.resid)
    var_seas_plus_resid = np.var(stl.seasonal + stl.resid)
    return float(max(0.0, 1.0 - var_remainder / var_seas_plus_resid))


def compute_seasonality_strength(train: pd.DataFrame, sample_ids: List[str]) -> pd.Series:

    return (
        pd.Series(
            {sid: get_stl_seasonality_strength(train.loc[sid].dropna().values.astype(float))
             for sid in sample_ids},
            name="stl_strength",
        )
        .sort_values(ascending=False)
    )

# код сгенерирован ии модель sonnet-4.6 промпт: напиши код подсчета влияния силы сезонности на smape в зависимости от модели для разных рядов используя существующие metrics.py
def compute_seasonality_effect(
    all_results: dict,
    sample_ids: List[str],
    seas_s: pd.Series,
    baseline: str = "lags",
    seas_models: List[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:

    if seas_models is None:
        seas_models = ["lags+seas", "lags+seas+fourier"]

    median_s = seas_s.median()
    strong = seas_s[seas_s >= median_s].index.tolist()
    weak   = seas_s[seas_s <  median_s].index.tolist()

    rows_s = []
    for name, res in all_results.items():
        rows_s.append({
            "model":        name,
            "SMAPE_strong": smape(res[res["sensor_id"].isin(strong)]["true_value"].values,
                                  res[res["sensor_id"].isin(strong)]["predicted_value"].values),
            "SMAPE_weak":   smape(res[res["sensor_id"].isin(weak)]["true_value"].values,
                                  res[res["sensor_id"].isin(weak)]["predicted_value"].values),
        })

    seas_effect_df = pd.DataFrame(rows_s).set_index("model").round(2)
    seas_effect_df["delta"] = (seas_effect_df["SMAPE_weak"] - seas_effect_df["SMAPE_strong"]).round(2)

    gain_per_series = {}
    for sid in sample_ids:
        base = smape(
            all_results[baseline][all_results[baseline]["sensor_id"] == sid]["true_value"].values,
            all_results[baseline][all_results[baseline]["sensor_id"] == sid]["predicted_value"].values,
        )
        best_seas = min(
            smape(all_results[m][all_results[m]["sensor_id"] == sid]["true_value"].values,
                  all_results[m][all_results[m]["sensor_id"] == sid]["predicted_value"].values)
            for m in seas_models
        )
        gain_per_series[sid] = base - best_seas

    return seas_effect_df, pd.Series(gain_per_series)
