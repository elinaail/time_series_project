"""
Глобальная модель CatBoost для прогнозирования временных рядов.

Ключевые проектные решения
---------------------------
Ряды M4 daily преимущественно финансовые (цены акций, индексы) — они
выраженно трендовые и приближены к случайному блужданию с недельной
сезонностью. Для корректной работы глобальной модели применяются:

1. Первые разности: 
    Обучение ведётся на dy[t] = y[t] - y[t-1] вместо y[t].
    Дифференцирование убирает тренд и делает каждый ряд приблизительно стационарным,
    что позволяет единой глобальной модели обобщаться на рядах с разными
    направлениями и масштабами тренда.

2. Нормировка разностей по каждому ряду:
    значения dy делятся на среднее абсолютное dy каждого ряда (MAD).
    Это устраняет различия масштаба между рядами, чтобы высоковолатильные ряды не доминировали обучение.

Стратегия прогноза (рекурсивная):
    Прогноз dy[T+1], затем dy[T+2], ..., затем восстановление уровней:
        y[T+k] = y[T] + dy[T+1] + dy[T+2] + ... + dy[T+k], где y[T] — последнее наблюдение обучающей выборки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.features import build_features, _get_feature_columns, _fourier_features, _calendar_features


class GlobalCatBoostForecaster:
    """
    Глобальная модель CatBoost, обучаемая на всех рядах для заданного варианта признаков.
    Работает в пространстве первых разностей для обработки нестационарных трендовых рядов.
    """

    def __init__(
        self,
        variant: dict,
        catboost_params: dict | None = None,
    ):
        self.variant = variant
        self.params = catboost_params or config.CATBOOST_PARAMS
        self._model: CatBoostRegressor | None = None
        self._feature_cols: list[str] = []
        self._cat_features: list[str] = []

    @property
    def name(self) -> str:
        return f"CatBoost_{self.variant['name']}"

    def fit(self, train_df: pd.DataFrame) -> "GlobalCatBoostForecaster":
        """
        Строит признаки на дифференцированном нормированном наборе данных и обучает CatBoost.

        Параметры
        ----------
        train_df: 
            данные для обучения.
        """
        df = train_df.sort_values(["series_id", "ds"]).copy()

        # первые разности внутри каждого ряда
        df["y"] = df.groupby("series_id")["y"].diff()
        df = df.dropna(subset=["y"])     # удаляем первую строку ряда (NaN после diff)

        # масштаб по ряду: среднее абсолютное значение разности (MAD)
        scale_map = df.groupby("series_id")["y"].transform(
            lambda s: s.abs().mean()
        )
        scale_map = scale_map.where(scale_map > 0, 1.0)
        df["y"] = df["y"] / scale_map

        df_feat = build_features(df, self.variant, target_col="y")

        self._feature_cols = _get_feature_columns(df_feat, self.variant)

        cat_candidates = ["day_of_week", "day_of_month", "month", "week_of_year", "quarter"]
        self._cat_features = [c for c in cat_candidates if c in self._feature_cols]

        X = df_feat[self._feature_cols].copy()
        y = df_feat["y"].to_numpy(dtype=float)

        cat_indices = [self._feature_cols.index(c) for c in self._cat_features]

        n_val = max(int(len(X) * 0.1), 1)
        train_pool_sub = Pool(
            data=X.iloc[:-n_val],
            label=y[:-n_val],
            cat_features=cat_indices if cat_indices else None,
        )
        val_pool = Pool(
            data=X.iloc[-n_val:],
            label=y[-n_val:],
            cat_features=cat_indices if cat_indices else None,
        )

        self._model = CatBoostRegressor(**self.params)
        self._model.fit(train_pool_sub, eval_set=val_pool)

        return self

    def predict_series(
        self,
        train_values: np.ndarray,
        h: int,
        start_date: pd.Timestamp | None = None,
    ) -> np.ndarray:
        """
        Рекурсивный прогноз на h шагов в исходном масштабе (уровни).

        Внутренно работает с первыми разностями, затем накапливает обратно:
            y[T+k] = y[T] + sum_{i=1}^{k} dy[T+i]

        Параметры
        ----------
        train_values: 
            Наблюдаемая история в исходном масштабе.
        h:
            Горизонт прогноза.
        start_date:
            Дата первого прогнозного шага.
        """
        # вычисляем масштаб по обучающим разностям
        train_diffs = np.diff(train_values)
        scale = float(np.mean(np.abs(train_diffs)))
        if scale == 0:
            scale = 1.0

        # история нормированных разностей
        history_diff = list(train_diffs / scale)

        if start_date is None:
            start_date = pd.Timestamp("2000-01-01")

        last_level = train_values[-1]
        t_offset = len(history_diff)   # позиция первой прогнозной разности в ряду разностей
        forecasted_diffs = []

        for step in range(h):
            row = self._build_row(
                history_diff,
                t_offset + step,
                start_date + pd.Timedelta(days=step),
            )
            if row is None:
                pred_diff = 0.0    # запасной вариант: изменений нет
            else:
                pred_diff = float(self._model.predict(row)[0])

            forecasted_diffs.append(pred_diff)
            history_diff.append(pred_diff)

        # восстанавливаем уровень через накопленные разности
        forecasted_diffs_unscaled = np.array(forecasted_diffs) * scale
        levels = last_level + np.cumsum(forecasted_diffs_unscaled)
        return levels

    def _build_row(
        self,
        history_diff: list[float],
        t: int,
        date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Строит однострочный DataFrame с признаками для позиции t в ряду разностей."""
        row = {}

        all_lags = []
        if self.variant.get("use_regular_lags"):
            all_lags += config.REGULAR_LAGS
        if self.variant.get("use_seasonal_lags"):
            all_lags += config.SEASONAL_LAGS

        for lag in sorted(set(all_lags)):
            if len(history_diff) < lag:
                return None
            row[f"lag_{lag}"] = history_diff[-lag]

        if self.variant.get("use_calendar"):
            row["day_of_week"] = date.dayofweek
            row["day_of_month"] = date.day
            row["month"] = date.month
            row["week_of_year"] = date.isocalendar()[1]
            row["quarter"] = date.quarter

        if self.variant.get("use_fourier"):
            for term in config.FOURIER_TERMS:
                period = term["period"]
                K = term["K"]
                tag = f"p{int(period)}" if period == int(period) else f"p{period}"
                for k in range(1, K + 1):
                    angle = 2 * np.pi * k * t / period
                    row[f"fourier_sin_{tag}_k{k}"] = np.sin(angle)
                    row[f"fourier_cos_{tag}_k{k}"] = np.cos(angle)

        # формируем DataFrame с правильным порядком столбцов
        df_row = pd.DataFrame([row])[self._feature_cols]
        return df_row
