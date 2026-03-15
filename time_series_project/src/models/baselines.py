"""
Базовые статистические модели: Naive, SeasonalNaive.

Локальные модели — обучаются независимо для каждого ряда.
"""
from __future__ import annotations
import numpy as np


class NaiveForecaster:
    """
    Наивный прогноз: повторяет последнее наблюдение для всех горизонтов.
    """
    name = "Naive"

    def fit(self, train: np.ndarray) -> "NaiveForecaster":
        self._last = train[-1]
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.full(h, self._last)


class SeasonalNaiveForecaster:
    """
    Сезонный наивный прогноз: повторяет последний сезонный цикл.
    """
    name = "SeasonalNaive"

    def __init__(self, period: int = 7):
        self.period = period

    def fit(self, train: np.ndarray) -> "SeasonalNaiveForecaster":
        self._season = train[-self.period:]
        return self

    def predict(self, h: int) -> np.ndarray:
        reps = (h // self.period) + 1
        return np.tile(self._season, reps)[:h]
