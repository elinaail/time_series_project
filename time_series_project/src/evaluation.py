"""
Метрики качества прогнозирования временных рядов.

Метрики:
    MAE
        Средняя абсолютная ошибка (зависит от масштаба, интуитивна)
    RMSE
        Корень среднеквадратичной ошибки (штрафует крупные выбросы)
    MASE
        Нормированная средняя абсолютная ошибка (шкала независима, в единицах сезонного Naive(1))
    sMAPE
        Симметричная средняя абсолютная процентная ошибка (стандарт соревнования M4)

Все метрики вычисляются в исходном (непреобразованном) масштабе.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def smape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0 + eps
    return float(np.mean(np.abs(actual - forecast) / denom) * 100)


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray,
    period: int = 1,
) -> float:
    """
    MASE с знаменателем сезонного Naive.

    знаменатель = mean(|train[t] - train[t - period]|) для t >= period
    """
    if len(train) <= period:
        return np.nan
    naive_errors = np.abs(train[period:] - train[:-period])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(actual - forecast)) / scale)


def compute_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray,
    period: int = 7,
) -> dict[str, float]:
    return {
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "sMAPE": smape(actual, forecast),
        "MASE": mase(actual, forecast, train, period=period),
    }


def aggregate_metrics(
    results: list[dict],
    metric_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    Агрегирует метрики по модели и горизонту.
    Вход:
        список словарей метрик (keys: series_id, model, horizon, MAE, RMSE, sMAPE, MASE).
    """
    if metric_keys is None:
        metric_keys = ["MAE", "RMSE", "sMAPE", "MASE"]

    df = pd.DataFrame(results)
    agg = (
        df.groupby(["model", "horizon"])[metric_keys]
        .mean()
        .reset_index()
        .round(4)
    )
    return agg
