from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "student-mat.csv"
TARGET = "G3"
EARLY_FEATURES = [
    "G1", "studytime", "absences", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "traveltime",
]
LATE_FEATURES = ["G1", "G2", *EARLY_FEATURES[1:]]
REQUIRED_COLUMNS = set(EARLY_FEATURES + ["G2", TARGET])


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the UCI math data (semicolon-delimited)."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    frame = pd.read_csv(data_path, sep=";")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    for column in ("G1", "G2", TARGET):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame
