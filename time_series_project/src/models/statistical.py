"""
Статистические модели: AutoETS и AutoTheta из statsforecast.

Локальные модели — обучаются независимо для каждого ряда через
библиотеку statsforecast (Nixtla) — Python-реализацию auto.ets и auto.theta из R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta


class AutoETSForecaster:
    """Обёртка statsforecast AutoETS для использования с одним рядом."""

    name = "AutoETS"

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self._model = None

    def fit(self, train: np.ndarray, series_id: str = "s1") -> "AutoETSForecaster":
        n = len(train)
        df = pd.DataFrame(
            {
                "unique_id": [series_id] * n,
                "ds": pd.date_range("2000-01-01", periods=n, freq="D"),
                "y": train,
            }
        )
        sf = StatsForecast(
            models=[AutoETS(season_length=self.season_length)],
            freq="D",
            n_jobs=1,
        )
        sf.fit(df)
        self._sf = sf
        self._series_id = series_id
        self._n = n
        return self

    def predict(self, h: int) -> np.ndarray:
        fc = self._sf.predict(h=h)
        return fc["AutoETS"].values


class AutoThetaForecaster:
    """Обёртка statsforecast AutoTheta для использования с одним рядом."""

    name = "AutoTheta"

    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def fit(self, train: np.ndarray, series_id: str = "s1") -> "AutoThetaForecaster":
        n = len(train)
        df = pd.DataFrame(
            {
                "unique_id": [series_id] * n,
                "ds": pd.date_range("2000-01-01", periods=n, freq="D"),
                "y": train,
            }
        )
        sf = StatsForecast(
            models=[AutoTheta(season_length=self.season_length)],
            freq="D",
            n_jobs=1,
        )
        sf.fit(df)
        self._sf = sf
        return self

    def predict(self, h: int) -> np.ndarray:
        fc = self._sf.predict(h=h)
        return fc["AutoTheta"].values
