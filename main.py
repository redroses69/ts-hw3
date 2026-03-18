import logging
import os

import config
from src.data import prepare_long_df
from src.experiments import run_baseline_experiments, run_horizon_experiments, run_main_experiments
from src.metrics import compute_seasonality_effect, compute_seasonality_strength, get_smape_table
from src.visualisations import plot_feature_forecasts, plot_horizon_effect, plot_seasonality_effect, plot_smape_comparison

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train, long_df = prepare_long_df(config.DATA_PATH, config.SAMPLE_IDS)

    log.info("Запуск бейзлайнов")
    baseline_results = run_baseline_experiments(long_df)
    log.info("Запуск основных экспериментов")
    main_results = run_main_experiments(long_df)
    all_results = {**baseline_results, **main_results}

    smape_df = get_smape_table(all_results, config.SAMPLE_IDS)
    smape_df.to_csv(os.path.join(config.OUTPUT_DIR, "smape_table.csv"))
    plot_feature_forecasts(
        all_results, long_df,
        sid=config.SAMPLE_IDS[0],
        model_names=["lags", "lags+seas", "lags+seas+fourier", "lags+seas+cal"],
        save_path=os.path.join(config.OUTPUT_DIR, "forecast.png"),
    )
    plot_smape_comparison(smape_df, config.SAMPLE_IDS,
                          save_path=os.path.join(config.OUTPUT_DIR, "smape_comparison.png"))

    seas_s = compute_seasonality_strength(train, config.SAMPLE_IDS)
    seas_effect_df, gain_s = compute_seasonality_effect(
        all_results, config.SAMPLE_IDS, seas_s,
        seas_models=["lags+seas", "lags+seas+fourier"],
    )

    strong = seas_s[seas_s >= seas_s.median()].index.tolist()
    plot_seasonality_effect(seas_effect_df, gain_s, seas_s, strong,
                            save_path=os.path.join(config.OUTPUT_DIR, "seasonality_effect.png"))

    horizon_results = run_horizon_experiments(long_df)
    
    plot_horizon_effect(
        horizon_results=horizon_results,
        horizons=config.HORIZONS,
        model_names=list(config.FEATURE_CONFIGS.keys()),
        seas_model_names=["lags+seas", "lags+seas+fourier", "lags+seas+cal"],
        save_path=os.path.join(config.OUTPUT_DIR, "horizon_true_effect.png"),
    )

    log.info("Пайплайн завершён успешно. Результаты сохранены в %s", config.OUTPUT_DIR)
