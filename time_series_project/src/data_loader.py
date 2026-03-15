"""
Загрузка и предобработка данных набора M4 daily (формат TSF).

Файл .tsf хранит временные ряды в формате:
    T<id>:<start_date>:<comma-separated-values>
    Несколько рядов могут размещаться на одной строке, разделённые пробелами.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# регулярное выражение для поиска записей рядов: T<id>:<date>:<values>
_SERIES_PATTERN = re.compile(
    r"T(\d+):([\d\-]+(?: \d{2}-\d{2}-\d{2})?):([0-9.,\-]+)"
)

# форматы дат, используемые в файле
_DATE_PATTERNS = [
    "%Y-%m-%d %H-%M-%S",
    "%Y-%m-%d",
]


def _parse_date(raw: str) -> pd.Timestamp | None:
    raw = raw.strip()
    for fmt in _DATE_PATTERNS:
        try:
            return pd.Timestamp(raw, )
        except Exception:
            pass
    try:
        return pd.to_datetime(raw, dayfirst=False)
    except Exception:
        return None


def load_tsf(path: str | Path) -> list[dict]:
    """
    Предобрабатывает файл M4 daily .tsf и возвращает список словарей с ключами:
        - series_id: str
        - start_date: pd.Timestamp или None- values: np.ndarray (float64)
    """
    raw_text = Path(path).read_text(encoding="utf-8", errors="replace")

    series_list = []
    for m in _SERIES_PATTERN.finditer(raw_text):
        sid = f"T{m.group(1)}"
        date_str = m.group(2)
        vals_str = m.group(3)

        try:
            values = np.array([float(v) for v in vals_str.split(",") if v.strip()])
        except ValueError:
            logger.warning("Невозможно разобрать значения ряда %s, пропуск.", sid)
            continue

        ts = _parse_date(date_str)
        series_list.append(
            {"series_id": sid, "start_date": ts, "values": values}
        )

    logger.info("Loaded %d series from %s", len(series_list), path)
    return series_list


def compute_seasonality_strength(values: np.ndarray, period: int = 7) -> float:
    """
    Вычисляет прокси "силы" сезонности:
        пик АФК на сезонном лаге относительно максимального АФК на несезонных лагах.
        Возвращает |ACF(period)|.
    """
    if len(values) < 2 * period + 1:
        return 0.0
    acf_vals = acf(values, nlags=period, fft=True)
    return float(abs(acf_vals[period]))


def sample_series(
    series_list: list[dict],
    n: int = config.N_SAMPLE,
    min_length: int = config.MIN_LENGTH,
    acf_threshold: float = config.ACF_LAG_THRESHOLD,
    period: int = config.SEASONAL_PERIOD,
    seed: int = config.RANDOM_SEED,
) -> list[dict]:
    """
    Фильтрует ряды по минимальной длине и силе сезонности и извлекает случайную выборку размером n.
    Добавляет ключ 'seasonality_strength' в каждый словарь.
    """
    rng = np.random.default_rng(seed)

    eligible = []
    for s in series_list:
        if len(s["values"]) < min_length:
            continue
        strength = compute_seasonality_strength(s["values"], period)
        if strength < acf_threshold:
            continue
        s = dict(s)  # копируем, чтобы не изменять оригинал
        s["seasonality_strength"] = strength
        eligible.append(s)

    logger.info(
        "Eligible series (len >= %d, ACF(7) >= %.2f): %d",
        min_length, acf_threshold, len(eligible)
    )

    if len(eligible) <= n:
        sample = eligible
    else:
        idx = rng.choice(len(eligible), size=n, replace=False)
        sample = [eligible[i] for i in idx]

    logger.info("Sampled %d series.", len(sample))
    return sample


def build_panel(
    series_list: list[dict],
    test_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Преобразует список словарей рядов в DataFrame для обучения/теста.
    """
    train_rows = []
    test_rows = []

    for s in series_list:
        sid = s["series_id"]
        values = s["values"]
        strength = s.get("seasonality_strength", 0.0)

        n = len(values)
        if n <= test_horizon:
            continue

        train_vals = values[: n - test_horizon]
        test_vals = values[n - test_horizon :]

        start = s["start_date"] if s["start_date"] is not None else pd.Timestamp("2000-01-01")
        dates = pd.date_range(start=start, periods=n, freq="D")
        train_dates = dates[: n - test_horizon]
        test_dates = dates[n - test_horizon :]

        for d, v in zip(train_dates, train_vals):
            train_rows.append(
                {"series_id": sid, "ds": d, "y": v, "seasonality_strength": strength}
            )
        for d, v in zip(test_dates, test_vals):
            test_rows.append(
                {"series_id": sid, "ds": d, "y": v, "seasonality_strength": strength}
            )

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df


def load_and_prepare(
    path: str | Path = config.DATA_PATH,
    test_horizon: int = 14,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн: загрузка - сэмплирование рядов - предобработка данных.

    Возвращает:
        sample_list, train_df, test_df
    """
    all_series = load_tsf(path)
    sample = sample_series(all_series)
    train_df, test_df = build_panel(sample, test_horizon=test_horizon)
    return sample, train_df, test_df
