"""
Feature Engineering для временных рядов.

Варианты:
    1. lags_only
        обычные краткосрочные лаги
    2. lags_seasonal
        лаги + сезонные лаги
    3. lags_calendar
        лаги + категориальные календарные признаки
    4. lags_fourier
        лаги + пары sin/cos ряда Фурье
    5. lags_seasonal_calendar
        лаги + сезонные лаги + категориальные календарные признаки
    6. lags_seasonal_fourier
        лаги + сезонные лаги + пары sin/cos ряда Фурье
    7. all_features
        все признаки комбинированно
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# вспомогательные функции построения признаков
def _lag_features(df: pd.DataFrame, lags: list[int], col: str = "y") -> pd.DataFrame:
    """Добавляет лаговые признаки."""
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("series_id")[col].shift(lag)
    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет категориальные календарные признаки."""
    df = df.copy()
    df["day_of_week"] = df["ds"].dt.dayofweek.astype("int8")
    df["day_of_month"] = df["ds"].dt.day.astype("int8")
    df["month"] = df["ds"].dt.month.astype("int8")
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype("int16")
    df["quarter"] = df["ds"].dt.quarter.astype("int8")
    return df


def _fourier_features(
    df: pd.DataFrame,
    fourier_terms: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Добавляет sin/cos-признаки ряда Фурье.

    Каждый элемент fourier_terms: {"period": float, "K": int}.
    """
    if fourier_terms is None:
        fourier_terms = config.FOURIER_TERMS

    df = df.copy()
    # порядковый номер внутри каждого ряда (начиная с 0) — без утечки информации из будущего
    df["_t"] = df.groupby("series_id").cumcount()

    for term in fourier_terms:
        period = term["period"]
        K = term["K"]
        for k in range(1, K + 1):
            angle = 2 * np.pi * k * df["_t"] / period
            tag = f"p{int(period)}" if period == int(period) else f"p{period}"
            df[f"fourier_sin_{tag}_k{k}"] = np.sin(angle)
            df[f"fourier_cos_{tag}_k{k}"] = np.cos(angle)

    df = df.drop(columns=["_t"])
    return df


def build_features(
    df: pd.DataFrame,
    variant: dict,
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Вычисляет признаки на основе времени.

    Параметры
    ----------
    df: DataFrame с колонками [series_id, ds, y, ...],
    variant:
        Cловарь с ключами из config.FEATURE_VARIANTS.
    target_col:
        Название целевого столбца.
    """
    df = df.sort_values(["series_id", "ds"]).copy()

    lags_to_add = []
    if variant.get("use_regular_lags"):
        lags_to_add += config.REGULAR_LAGS
    if variant.get("use_seasonal_lags"):
        lags_to_add += config.SEASONAL_LAGS

    if lags_to_add:
        df = _lag_features(df, sorted(set(lags_to_add)), col=target_col)

    if variant.get("use_calendar"):
        df = _calendar_features(df)

    if variant.get("use_fourier"):
        df = _fourier_features(df)

    # удаляем строки с NaN в признаках (так как лаги создают пропуски)
    feature_cols = _get_feature_columns(df, variant)
    df = df.dropna(subset=feature_cols)
    return df


def _get_feature_columns(df: pd.DataFrame, variant: dict) -> list[str]:
    cols = []
    cols += [c for c in df.columns if c.startswith("lag_")]
    if variant.get("use_calendar"):
        for c in ["day_of_week", "day_of_month", "month", "week_of_year", "quarter"]:
            if c in df.columns:
                cols.append(c)
    if variant.get("use_fourier"):
        cols += [c for c in df.columns if c.startswith("fourier_")]
    return cols


def get_feature_names(variant: dict) -> list[str]:
    """
    Возвращает названия столбцов признаков (необходимо для будующего отчета)
    """
    names = []
    if variant.get("use_regular_lags"):
        names += [f"lag_{l}" for l in config.REGULAR_LAGS]
    if variant.get("use_seasonal_lags"):
        names += [f"lag_{l}" for l in config.SEASONAL_LAGS]
    if variant.get("use_calendar"):
        names += ["day_of_week", "day_of_month", "month", "week_of_year", "quarter"]
    if variant.get("use_fourier"):
        for term in config.FOURIER_TERMS:
            period = term["period"]
            tag = f"p{int(period)}" if period == int(period) else f"p{period}"
            for k in range(1, term["K"] + 1):
                names += [f"fourier_sin_{tag}_k{k}", f"fourier_cos_{tag}_k{k}"]
    return names


def prepare_Xy(
    df_feat: pd.DataFrame,
    variant: dict,
    target_col: str = "y",
) -> tuple[pd.DataFrame, pd.Series]:
    """Разбивает DataFrame на X (признаки) и y (целевая переменная)."""
    feature_cols = _get_feature_columns(df_feat, variant)
    X = df_feat[feature_cols]
    y = df_feat[target_col]
    return X, y
