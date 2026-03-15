"""
Глобальная конфигурация эксперимента по првоерке гипотезы.
Все параметры эксперимента вынесены для обеспечения воспроизводимости и легкого конфигурирования проекта.
"""
import os

# пути к файлам
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "m4_daily_dataset", "m4_daily_dataset.tsf")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# выборка 
RANDOM_SEED = 42
N_SAMPLE = 150          # количество рядов в выборке
MIN_LENGTH = 700        # минимальная длина ряда для включения в выборку
ACF_LAG_THRESHOLD = 0.2 # минимальный |ACF(lag=7)| для признания ряда сезонным

# параметры временных рядов
SEASONAL_PERIOD = 7          # недельная сезонность в дневных данных
FORECAST_HORIZONS = [7, 14]  # короткий и стандартный горизонты прогноза M4-daily

# признаки
REGULAR_LAGS = [1, 2, 3, 4, 5, 6]  # (краткосрочные) лаги
SEASONAL_LAGS = [7, 14, 21, 28]    # cезонные лаги (кратные сезонному периоду)

# Члены ряда Фурье: список пар (period, K) (period=7  → недельный цикл, K=3 гармоники)
# period=365.25 → годовой цикл, K=5 гармоник
FOURIER_TERMS = [
    {"period": 7, "K": 3},
    {"period": 365.25, "K": 5},
]

# вариантов признаков для моделирования. Например, "lags_only" обучает модели, используя только признаки на лагах
FEATURE_VARIANTS = [
    {
        "name": "lags_only",
        "label": "Только обычные лаги",
        "use_regular_lags": True,
        "use_seasonal_lags": False,
        "use_calendar": False,
        "use_fourier": False,
    },
    {
        "name": "lags_seasonal",
        "label": "Лаги + сезонные лаги",
        "use_regular_lags": True,
        "use_seasonal_lags": True,
        "use_calendar": False,
        "use_fourier": False,
    },
    {
        "name": "lags_calendar",
        "label": "Лаги + календарные признаки",
        "use_regular_lags": True,
        "use_seasonal_lags": False,
        "use_calendar": True,
        "use_fourier": False,
    },
    {
        "name": "lags_fourier",
        "label": "Лаги + Фурье-признаки",
        "use_regular_lags": True,
        "use_seasonal_lags": False,
        "use_calendar": False,
        "use_fourier": True,
    },
    {
        "name": "lags_seasonal_calendar",
        "label": "Лаги + сезонные лаги + календарные признаки",
        "use_regular_lags": True,
        "use_seasonal_lags": True,
        "use_calendar": True,
        "use_fourier": False,
    },
    {
        "name": "lags_seasonal_fourier",
        "label": "Лаги + сезонные лаги + Фурье-признаки",
        "use_regular_lags": True,
        "use_seasonal_lags": True,
        "use_calendar": False,
        "use_fourier": True,
    },
    {
        "name": "all_features",
        "label": "Все признаки",
        "use_regular_lags": True,
        "use_seasonal_lags": True,
        "use_calendar": True,
        "use_fourier": True,
    },
]

# гиперпараметры CatBoost
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "early_stopping_rounds": 50,
    "thread_count": -1,
}

# statsforecast (бейзлайны)
STATSFORECAST_N_JOBS = -1 # использовать все ядра для обучения бейзлайнов statsforecast
