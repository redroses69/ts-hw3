from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

"""
код визуализации сгенерирован ии. модель sonnet-4.6
промпт: напиши код визуализации эффекта метрик из файла metrics.py визуализации должны отображать cравнение нескольких 
вариантов признаков — только обычные лаги; лаги +
сезонные лаги; лаги + категориальные календарные признаки; лаги +
фурье-признаки и не менее двух их комбинаций.
А также зависимость эффекта от силы сезонности и от горизонта
прогнозирования.
"""

def plot_smape_comparison(
    smape_df: pd.DataFrame,
    sample_ids: List[str],
    save_path: str = "smape_comparison.png",
) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    colors = ["steelblue" if "lags" in m.lower() else "lightgray" for m in smape_df.index]
    smape_df["SMAPE_overall"].plot(kind="barh", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title("Overall SMAPE по моделям", fontsize=12)
    axes[0].set_xlabel("SMAPE %")
    axes[0].invert_yaxis()

    for bar in axes[0].patches:
        axes[0].text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}%", va="center", fontsize=9,
        )

    heatmap_data = smape_df[sample_ids]
    im = axes[1].imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn_r")
    axes[1].set_xticks(range(len(sample_ids)))
    axes[1].set_xticklabels(sample_ids, rotation=45, ha="right")
    axes[1].set_yticks(range(len(smape_df)))
    axes[1].set_yticklabels(smape_df.index)
    axes[1].set_title("SMAPE по рядам", fontsize=12)
    plt.colorbar(im, ax=axes[1], label="SMAPE %")
    for i in range(len(smape_df)):
        for j in range(len(sample_ids)):
            axes[1].text(
                j, i, f"{heatmap_data.values[i, j]:.1f}",
                ha="center", va="center", fontsize=8, color="black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")


def plot_seasonality_effect(
    seas_effect_df: pd.DataFrame,
    gain_s: pd.Series,
    seas_s: pd.Series,
    strong: List[str],
    save_path: str = "seasonality_effect.png",
) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x = np.arange(len(seas_effect_df))
    w = 0.35
    axes[0].barh(x - w / 2, seas_effect_df["SMAPE_strong"], w,
                 label="Сильная сезонность", color="mediumseagreen")
    axes[0].barh(x + w / 2, seas_effect_df["SMAPE_weak"], w,
                 label="Слабая сезонность", color="tomato")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(seas_effect_df.index)
    axes[0].set_xlabel("SMAPE %")
    axes[0].set_title("SMAPE: сильная vs слабая сезонность", fontsize=12)
    axes[0].legend()
    axes[0].invert_yaxis()

    colors_g = ["mediumseagreen" if sid in strong else "tomato" for sid in seas_s.index]
    axes[1].bar(seas_s.index, gain_s[seas_s.index], color=colors_g, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(
        "Прирост от сезонных фичей vs сила сезонности\n(зел = сильная, кр = слабая)",
        fontsize=11,
    )
    axes[1].set_xlabel("Ряд")
    axes[1].set_ylabel("ΔSMAPE")
    for i, (sid, val) in enumerate(gain_s[seas_s.index].items()):
        axes[1].text(i, val + 0.05, f"{val:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")

# код сгенерирован модель sonnet-4.6 промпт добавь отрисовку прогноза 3 рядов лучшей моделью для этого ряда  
def plot_feature_forecasts(
    all_results: Dict,
    long_df: pd.DataFrame,
    sid: str,
    model_names: List[str],
    history_window: int = 52,
    save_path: str = "forecast.png",
) -> None:

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4), sharey=True)

    for ax, name, color in zip(axes, model_names, colors):
        res = all_results[name]
        series_res = res[res["sensor_id"] == sid].sort_values("timestamp")
        last_fold = series_res["fold"].max()
        series_res = series_res[series_res["fold"] == last_fold]

        first_pred_ts = series_res["timestamp"].min()
        history = long_df[long_df["sensor_id"] == sid].sort_values("timestamp")
        history = history[history["timestamp"] < first_pred_ts].tail(history_window)

        ax.plot(history["timestamp"], history["value"], color="steelblue", label="История")
        ax.plot(series_res["timestamp"], series_res["true_value"], color="black", label="Факт")
        ax.plot(series_res["timestamp"], series_res["predicted_value"],
                color=color, linestyle="--", label="Прогноз")
        ax.set_title(f"{sid}  |  {name}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")


def plot_horizon_effect(
    horizon_results: Dict,
    horizons: List[int],
    model_names: List[str],
    seas_model_names: List[str],
    save_path: str = "horizon_true_effect.png",
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    colors = plt.cm.tab10.colors
    for idx, name in enumerate(model_names):
        vals = [horizon_results[(h, name)] for h in horizons]
        axes[0].plot(
            horizons, vals,
            label=name,
            color="silver" if name == "naive" else colors[idx % len(colors)],
            linestyle="--" if name == "naive" else "-",
            linewidth=2 if "seas" in name else 1.2,
            marker="o", markersize=5,
        )
    axes[0].set_xlabel("Горизонт прогнозирования (недели)")
    axes[0].set_ylabel("SMAPE %")
    axes[0].set_title("SMAPE vs горизонт прогнозирования", fontsize=12)
    axes[0].set_xticks(horizons)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    x = np.arange(len(horizons))
    w = 0.25
    bar_colors = ["steelblue", "mediumseagreen", "mediumpurple"]
    for i, seas_name in enumerate(seas_model_names):
        gains = [horizon_results[(h, "lags")] - horizon_results[(h, seas_name)] for h in horizons]
        axes[1].bar(x + i * w, gains, w, label=seas_name,
                    color=bar_colors[i], edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x + w)
    axes[1].set_xticklabels([f"h={h}" for h in horizons])
    axes[1].set_ylabel("ΔSMAPE")
    axes[1].set_title("Прирост от сезонных фичей по горизонтам", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
