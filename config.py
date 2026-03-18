from typing import Dict, List # noqa: F401

DATA_PATH: str = "data/m4_weekly_train.csv"
OUTPUT_DIR: str = "results"

SAMPLE_IDS: List[str] = [
    "W234", "W247", "W7", "W27", "W30",
    "W31", "W237", "W6", "W56", "W243",
]


FREQ: str = "W"
MODEL_HORIZON: int = 1    
HISTORY_ALL: int= 54   
N_LAGS: int = 8

SEASONAL_PERIOD: int = 52
N_SEASONAL_LAGS: int = 3
FOURIER_ORDER:int = 3

HORIZON: int = 13             
START_TRAIN: int = HISTORY_ALL * 3 
STEP:int = HORIZON * 4

HORIZONS: List[int] = [4, 8, 13]

FEATURE_CONFIGS: Dict[str, List] = {
    "pure catboost":     [],
    "lags":              ["lags"],
    "lags+seas":         ["lags", "seasonal_lags"],
    "lags+calendar":     ["lags", "calendar"],
    "lags+fourier":      ["lags", "fourier"],
    "lags+seas+fourier": ["lags", "seasonal_lags", "fourier"],
    "lags+seas+cal":     ["lags", "seasonal_lags", "calendar"],
}

