import pandas as pd
from statsforecast.models import AutoETS, AutoTheta, Naive, SeasonalNaive

import config
from src.metrics import smape
from src.models import CatBoostRecursive, StatsforecastModel
from src.validation import expanding_window_validation


def get_model(feature_modes, horizon: int):

    if feature_modes is None:
        return StatsforecastModel(Naive(), freq=config.FREQ, horizon=horizon)
    return CatBoostRecursive(
        model_horizon=config.MODEL_HORIZON,
        history=config.HISTORY_ALL,
        horizon=horizon,
        freq=config.FREQ,
        n_lags=config.N_LAGS,
        seasonal_period=config.SEASONAL_PERIOD,
        n_seasonal_lags=config.N_SEASONAL_LAGS,
        fourier_order=config.FOURIER_ORDER,
        feature_modes=feature_modes,
    )


def get_params(long_df: pd.DataFrame, horizon: int) -> dict:
    return dict(
        data=long_df,
        horizon=horizon,
        history=config.HISTORY_ALL,
        start_train_size=config.START_TRAIN,
        step_size=config.STEP,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    )


def run_main_experiments(long_df: pd.DataFrame) -> dict:

    all_results = {}

    for name, modes in config.FEATURE_CONFIGS.items():

        all_results[name] = expanding_window_validation(
            model=get_model(modes, config.HORIZON),
            **get_params(long_df, config.HORIZON),
        )

    return all_results


def run_baseline_experiments(long_df: pd.DataFrame) -> dict:
    baseline_models = {
        "Naive": Naive(),
        "Seasonal Naive": SeasonalNaive(season_length=config.SEASONAL_PERIOD),
        "Auto.Theta": AutoTheta(season_length=config.SEASONAL_PERIOD),
        "Auto.ETS": AutoETS(season_length=config.SEASONAL_PERIOD),
    }

    results = {}
    for name, sf_model in baseline_models.items():
        results[name] = expanding_window_validation(
            model=StatsforecastModel(sf_model, freq=config.FREQ, horizon=config.HORIZON),
            **get_params(long_df, config.HORIZON),
        )
    return results


def run_horizon_experiments(long_df: pd.DataFrame) -> dict:

    horizon_results = {}

    for h in config.HORIZONS:
        for name, modes in config.FEATURE_CONFIGS.items():
 
            res = expanding_window_validation(
                model=get_model(modes, h),
                **get_params(long_df, h),
            )
            
            horizon_results[(h, name)] = smape(res["true_value"].values, res["predicted_value"].values)

    horizon_df = (
        pd.Series(horizon_results)
        .rename_axis(["horizon", "model"])
        .unstack("horizon")
        .round(2)
    )
    print("\nSMAPE по горизонтам и моделям:\n")
    print(horizon_df.to_string())
    return horizon_results
