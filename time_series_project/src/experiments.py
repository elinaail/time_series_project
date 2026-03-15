"""
Запуск экспериментов

Пайплайн:
    1. Загрузка и выборка набора M4 daily.
    2. Запуск локальных статистических бейзлайнов (Naive, SeasonalNaive, AutoETS, AutoTheta)
        — Оценка для каждого ряда и горизонта.
    3. Для каждого варианта признаков CatBoost — обучение глобальной модели
        - Рекурсивная оценка на каждом тестовом ряде.
    4. Агрегация метрик и сохранение результатов в results/.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import load_and_prepare
from src.evaluation import compute_metrics, aggregate_metrics
from src.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from src.models.statistical import AutoETSForecaster, AutoThetaForecaster
from src.models.catboost_model import GlobalCatBoostForecaster

logger = logging.getLogger(__name__)


def _eval_local_model(model, train: np.ndarray, test: np.ndarray, h: int, sid: str) -> list[dict]:
    """Обучает и оценивает локальную модель (посерийную) для заданного горизонта."""
    records = []
    train_h = train  # use full train for fitting

    try:
        if hasattr(model, "fit"):
            if isinstance(model, (AutoETSForecaster, AutoThetaForecaster)):
                model.fit(train_h, series_id=sid)
            else:
                model.fit(train_h)
        fc = model.predict(h)
        actual = test[:h]
        metrics = compute_metrics(actual, fc, train_h, period=config.SEASONAL_PERIOD)
        metrics.update({"model": model.name, "horizon": h, "series_id": sid})
        records.append(metrics)
    except Exception as e:
        logger.warning("Error in %s for %s h=%d: %s", model.name, sid, h, e)

    return records


def run_baselines(
    sample_list: list[dict],
    horizons: list[int],
) -> list[dict]:
    """Запускает Naive, SeasonalNaive, AutoETS, AutoTheta на всех рядах."""
    records = []
    models_cls = [
        lambda: NaiveForecaster(),
        lambda: SeasonalNaiveForecaster(period=config.SEASONAL_PERIOD),
        lambda: AutoETSForecaster(season_length=config.SEASONAL_PERIOD),
        lambda: AutoThetaForecaster(season_length=config.SEASONAL_PERIOD),
    ]

    max_h = max(horizons)

    for s in tqdm(sample_list, desc="Baselines"):
        values = s["values"]
        if len(values) <= max_h:
            continue
        train = values[: len(values) - max_h]
        test = values[len(values) - max_h :]
        sid = s["series_id"]
        strength = s.get("seasonality_strength", 0.0)

        for mk in models_cls:
            model = mk()
            for h in horizons:
                recs = _eval_local_model(model, train, test, h, sid)
                for r in recs:
                    r["seasonality_strength"] = strength
                records.extend(recs)

    return records


def run_catboost_experiments(
    sample_list: list[dict],
    train_df: pd.DataFrame,
    horizons: list[int],
    variants: list[dict],
) -> list[dict]:
    """Обучает и оценивает одну модель CatBoost на каждый вариант признаков."""
    records = []
    max_h = max(horizons)

    for variant in variants:
        logger.info("Training CatBoost variant: %s", variant["name"])
        t0 = time.time()

        model = GlobalCatBoostForecaster(variant=variant)
        try:
            model.fit(train_df)
        except Exception as e:
            logger.error("Failed to train %s: %s", variant["name"], e)
            continue

        logger.info("  Trained in %.1fs, evaluating …", time.time() - t0)

        for s in tqdm(sample_list, desc=f"CB {variant['name']}", leave=False):
            values = s["values"]
            if len(values) <= max_h:
                continue

            train_vals = values[: len(values) - max_h]
            test_vals = values[len(values) - max_h :]
            sid = s["series_id"]
            strength = s.get("seasonality_strength", 0.0)
            start_date = s["start_date"]
            if start_date is None:
                start_date = pd.Timestamp("2000-01-01")
            forecast_start = start_date + pd.Timedelta(days=len(train_vals))

            try:
                fc_full = model.predict_series(
                    train_values=train_vals,
                    h=max_h,
                    start_date=forecast_start,
                )
            except Exception as e:
                logger.warning("Prediction error %s %s: %s", variant["name"], sid, e)
                continue

            for h in horizons:
                actual = test_vals[:h]
                fc = fc_full[:h]
                metrics = compute_metrics(actual, fc, train_vals, period=config.SEASONAL_PERIOD)
                metrics.update(
                    {
                        "model": model.name,
                        "horizon": h,
                        "series_id": sid,
                        "seasonality_strength": strength,
                    }
                )
                records.append(metrics)

    return records


def run_all_experiments(
    data_path: str | Path = config.DATA_PATH,
    horizons: list[int] | None = None,
    variants: list[dict] | None = None,
    results_dir: str | Path = config.RESULTS_DIR,
) -> pd.DataFrame:
    """
    Полный пайплайн эксперимента.

    Возвращает
    -------
    DataFrame с посерийными, помодельными, погоризонтными метриками.
    """
    if horizons is None:
        horizons = config.FORECAST_HORIZONS
    if variants is None:
        variants = config.FEATURE_VARIANTS

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. загрузка + предобработка данных 
    logger.info("Loading data from %s …", data_path)
    sample_list, train_df, test_df = load_and_prepare(data_path, test_horizon=max(horizons))
    logger.info(
        "Panel: %d train rows, %d test rows, %d series",
        len(train_df), len(test_df), len(sample_list),
    )

    # сохранение метаданных выборки
    meta = [
        {"series_id": s["series_id"], "seasonality_strength": s.get("seasonality_strength", 0.0),
         "length": len(s["values"])}
        for s in sample_list
    ]
    pd.DataFrame(meta).to_csv(results_dir / "sample_meta.csv", index=False)

    # обучаем и оцениваем baseline модели 
    logger.info("Запуск бейзлайнов …")
    baseline_records = run_baselines(sample_list, horizons)
    logger.info("Записей бейзлайнов: %d", len(baseline_records))

    # обучаем и оцениваем CatBoost на разных вариантах признаков 
    logger.info("Запуск экспериментов CatBoost …")
    cb_records = run_catboost_experiments(sample_list, train_df, horizons, variants)
    logger.info("Записей CatBoost: %d", len(cb_records))

    # объединяем и сохраняем результаты
    all_records = baseline_records + cb_records
    raw_df = pd.DataFrame(all_records)
    raw_df.to_csv(results_dir / "raw_metrics.csv", index=False)

    # агрегированные результаты 
    agg_df = aggregate_metrics(all_records)
    agg_df.to_csv(results_dir / "aggregated_metrics.csv", index=False)

    logger.info("Результаты сохранены в %s", results_dir)
    return raw_df
