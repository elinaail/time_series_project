#!/usr/bin/env python
"""
Точка входа для запуска экспериментов согласно заданию.
    - Все параметры управляются через config.py
    - Результаты сохраняются в results/
"""
import config
import logging
import sys
import time
from pathlib import Path
from src.experiments import run_all_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/experiment.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# директория results должна существовать до инициализации файлового обработчика
Path("results").mkdir(exist_ok=True)

if __name__ == "__main__":
    logger.info("Размер выборки: %d", config.N_SAMPLE)
    logger.info("Горизонты: %s", config.FORECAST_HORIZONS)
    logger.info("Сезон. период: %d", config.SEASONAL_PERIOD)
    logger.info("Используемый набор признаков: %d", len(config.FEATURE_VARIANTS))

    t0 = time.time()
    raw_df = run_all_experiments()

    elapsed = time.time() - t0
    logger.info("Эксперимент завершён за %.1f секунд.", elapsed)

    # сводная талбица результатов
    from src.evaluation import aggregate_metrics
    agg = aggregate_metrics(raw_df.to_dict(orient="records"))
    print("\nАгрегированные метрики (среднее по всем рядам)")
    print(agg.to_string(index=False))
