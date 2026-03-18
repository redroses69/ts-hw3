import logging
import catboost as cb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive

log = logging.getLogger(__name__)

from src.feature_generation import get_features_df_and_targets
from src.index_slicing import features__test_idx, features_targets__train_idx


class BaseModel:
    """Базовый класс модели."""

    def __init__(self):
        raise NotImplementedError

    def fit(
        self, train_data, val_data, id_col="ts_id", timestamp_col="timestamp", value_col="value"
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns: None
        """
        raise NotImplementedError

    def predict(self, test_data, id_col="ts_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].
        """
        raise NotImplementedError


class StatsforecastModel(BaseModel):
    """Модель, использующая библиотеку statsforecast для прогнозирования."""

    def __init__(self, model, freq: str, horizon: int):
        """Инициализация модели.

        Args:
            - model: экземпляр модели из библиотеки statsforecast.
            - freq: частота временного ряда (например, 'H' для почасовых данных).
            - horizon: общий горизонт прогнозирования.

        """
        self.model = model
        self.freq = freq
        self.horizon = horizon

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.

        """
        log.info("Fit StatsforecastModel: train=%d val=%d", len(train_data), len(val_data))
        # Объединяем тренировочные и валидационные данные
        combined_data = pd.concat([train_data, val_data])
        # Удаялем дубликаты, которые образовались после объединения
        # так как val_data начинается с history точек из конца train_data
        combined_data = combined_data.drop_duplicates(subset=[id_col, timestamp_col], keep="last")

        # Преобразуем данные в формат, необходимый для StatsForecast
        sf = StatsForecast(models=[self.model], freq=self.freq)
        self.sf = sf.fit(
            combined_data.rename(
                columns={id_col: "unique_id", timestamp_col: "ds", value_col: "y"}
            )
        )

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
            - test_data: DataFrame с тестовыми данными.

        Returns:
            - predictions: DataFrame с предсказанными значениями со столбцами
                [id_col, timestamp_col, 'predicted_value'].

        """
        forecasts = self.sf.predict(h=self.horizon)

        # Преобразуем прогнозы обратно в исходный формат
        pred_column = [col for col in forecasts.columns if col not in ["unique_id", "ds"]][0]
        predictions = forecasts[["unique_id", "ds", pred_column]].rename(
            columns={"unique_id": id_col, "ds": timestamp_col, pred_column: "predicted_value"}
        )

        return predictions


class CatBoostRecursive(BaseModel):
    """Модель CatBoost с рекурсивной стратегией прогнозирования."""

    def __init__(
        self,
        model_horizon: int,
        history: int,
        horizon: int,
        freq: str,
        feature_modes: list = None,
        n_lags: int = 8,
        seasonal_period: int = 52,
        n_seasonal_lags: int = 3,
        fourier_order: int = 3,
    ):
        """Инициализация модели.

        Args:
            - model_horizon:    Горизонт прогнозирования модели.
            - history:          Размер окна истории. Должен быть >= seasonal_period +
                                n_seasonal_lags - 1 при использовании 'seasonal_lags'.
            - horizon:          Общий горизонт прогнозирования.
            - freq:             Частота временного ряда.
            - feature_modes:    Список режимов генерации признаков. Допустимые значения:
                                'lags', 'seasonal_lags', 'calendar', 'fourier'.
                                По умолчанию ['lags'].
            - n_lags:           Число обычных лагов (режим 'lags').
            - seasonal_period:  Период сезонности (52 для недельных данных).
            - n_seasonal_lags:  Число сезонных лагов (режим 'seasonal_lags').
            - fourier_order:    Число гармоник Фурье (режим 'fourier').

        """
        self.model_horizon = model_horizon
        self.history = history
        self.horizon = horizon
        self.freq = freq
        self.feature_modes = feature_modes if feature_modes is not None else ["lags"]
        self.n_lags = n_lags
        self.seasonal_period = seasonal_period
        self.n_seasonal_lags = n_seasonal_lags
        self.fourier_order = fourier_order
        self.model = None

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        """
        log.info(
            "Fit CatBoostRecursive: modes=%s",
            self.feature_modes,
        )
        # Формируем индексы для признаков и таргетов
        train_features_idx, train_targets_idx = features_targets__train_idx(
            id_column=train_data[id_col],
            series_length=len(train_data),
            model_horizon=self.model_horizon,
            history_size=self.history,
        )
        val_features_idx, val_targets_idx = features_targets__train_idx(
            id_column=val_data[id_col],
            series_length=len(val_data),
            model_horizon=self.model_horizon,
            history_size=self.history,
        )
        # Генерируем признаки и таргеты
        feat_kwargs = dict(
            id_column=id_col,
            date_column=timestamp_col,
            target_column=value_col,
            feature_modes=self.feature_modes,
            n_lags=self.n_lags,
            seasonal_period=self.seasonal_period,
            n_seasonal_lags=self.n_seasonal_lags,
            fourier_order=self.fourier_order,
        )
        train_features, train_targets, categorical_features_idx = get_features_df_and_targets(
            train_data, train_features_idx, train_targets_idx, **feat_kwargs
        )
        val_features, val_targets, _ = get_features_df_and_targets(
            val_data, val_features_idx, val_targets_idx, **feat_kwargs
        )
        
        cb_model = cb.CatBoostRegressor(
            loss_function="MultiRMSE",
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100,
            iterations=1500,
            cat_features=categorical_features_idx,
        )
        train_dataset = cb.Pool(
            data=train_features, label=train_targets, cat_features=categorical_features_idx
        )
        eval_dataset = cb.Pool(
            data=val_features, label=val_targets, cat_features=categorical_features_idx
        )
        cb_model.fit(
            train_dataset,
            eval_set=eval_dataset,
            use_best_model=True,
            plot=False,
        )
        self.model = cb_model

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """
        Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].

        """
        # Последовательно прогнозируем значения
        steps = self.horizon // self.model_horizon

        for step in range(steps):
            print(f"Шаг прогнозирования {step + 1} из {steps}")

            test_features_idx, target_features_idx = features__test_idx(
                id_column=test_data[id_col],
                series_length=len(test_data),
                model_horizon=self.model_horizon,
                history_size=self.history + step * self.model_horizon,
            )
            test_features_idx = test_features_idx[:, step:]

            test_features, _, _ = get_features_df_and_targets(
                test_data,
                test_features_idx,
                target_features_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
                feature_modes=self.feature_modes,
                n_lags=self.n_lags,
                seasonal_period=self.seasonal_period,
                n_seasonal_lags=self.n_seasonal_lags,
                fourier_order=self.fourier_order,
            )

            test_preds = self.model.predict(test_features)

            # Заполняем предсказаниями соответствующие места в истории
            test_data.iloc[
                target_features_idx.flatten(),  # Берем только последние индексы таргетов
                test_data.columns.get_loc(value_col),
            ] = test_preds.reshape(-1, 1)

        # Оставляем в итоговом датафрейме только нужные строки и столбцы
        first_test_date = np.sort(np.unique(test_data[timestamp_col]))[self.history]
        test_data = test_data[test_data[timestamp_col] >= first_test_date]
        test_data = test_data.rename(columns={value_col: "predicted_value"})

        return test_data[[id_col, timestamp_col, "predicted_value"]].reset_index(drop=True)


class CatBoostDirect(BaseModel):
    """Модель CatBoost с прямой стратегией прогнозирования."""

    def __init__(self, model_horizon: int, history: int, horizon: int, freq: str):
        """Инициализация модели.

        Args:
            - model_horizon: Горизонт прогнозирования модели.
            - history: Размер окна истории.
            - horizon: Общий горизонт прогнозирования.
            - freq: Частота временного ряда.

        """
        self.model_horizon = model_horizon
        self.history = history
        self.horizon = horizon
        self.freq = freq
        self.models = []

    def fit(self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
        ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        """

        steps = self.horizon // self.model_horizon
        self.models = []

        for step in range(steps):
            print(f"Обучение {step + 1} из {steps}")

            train_features_idx, train_targets_idx = features_targets__train_idx(
                id_column=train_data[id_col],
                series_length=len(train_data) - step * self.model_horizon,
                model_horizon=self.model_horizon,
                history_size=self.history,
            )

            train_features_idx += step * self.model_horizon
            train_targets_idx += step * self.model_horizon

            val_features_idx, val_targets_idx = features_targets__train_idx(
                id_column=val_data[id_col],
                series_length=len(val_data) - step * self.model_horizon,
                model_horizon=self.model_horizon,
                history_size=self.history,
            )

            val_features_idx += step * self.model_horizon
            val_targets_idx += step * self.model_horizon

            train_features, train_targets, cat_idx = get_features_df_and_targets(
                train_data,
                train_features_idx,
                train_targets_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
            )

            val_features, val_targets, _ = get_features_df_and_targets(
                val_data,
                val_features_idx,
                val_targets_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
            )

            model = cb.CatBoostRegressor(
                loss_function="MultiRMSE",
                random_seed=42,
                verbose=100,
                early_stopping_rounds=100,
                iterations=1500,
                cat_features=cat_idx,
            )

            model.fit(
                cb.Pool(train_features, train_targets, cat_features=cat_idx),
                eval_set=cb.Pool(val_features, val_targets, cat_features=cat_idx),
                use_best_model=True,
                plot=False,
            )

            self.models.append(model)

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].

        """

        test_data = test_data.copy()
        steps = self.horizon // self.model_horizon

        for step, model in enumerate(self.models):

            print(f"Direct прогноз {step + 1} из {steps}")

            test_features_idx, test_targets_idx = features__test_idx(
                id_column=test_data[id_col],
                series_length=len(test_data) - step * self.model_horizon,
                model_horizon=self.model_horizon,
                history_size=self.history,
            )

            test_features_idx += step * self.model_horizon
            test_targets_idx += step * self.model_horizon

            test_features, _, _ = get_features_df_and_targets(
                test_data,
                test_features_idx,
                test_targets_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
            )

            preds = model.predict(test_features)

            test_data.iloc[
                test_targets_idx.flatten(),
                test_data.columns.get_loc(value_col),
            ] = preds.reshape(-1, 1)

        first_test_date = np.sort(np.unique(test_data[timestamp_col]))[self.history]
        test_data = test_data[test_data[timestamp_col] >= first_test_date]
        test_data = test_data.rename(columns={value_col: "predicted_value"})

        return test_data[[id_col, timestamp_col, "predicted_value"]].reset_index(drop=True)


