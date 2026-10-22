from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_model(path: str | Path) -> Any:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run python -m src.train first.")
    return joblib.load(model_path)


def predict_grades(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Return presentation-bounded grade estimates on UCI's 0–20 scale."""
    predictions = np.asarray(model.predict(frame), dtype=float)
    return np.clip(predictions, 0, 20)


def risk_category(grade: float, threshold: float = 10.0) -> str:
    if not 0 <= threshold <= 20:
        raise ValueError("Threshold must be between 0 and 20.")
    return "Review recommended" if grade < threshold else "At/above illustrative threshold"


def validate_batch(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Validate model columns and numeric ranges; return a clean feature-only copy."""
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns) - set(features))
    if extra:
        raise ValueError(f"Unsupported columns: {', '.join(extra)}. Remove names, IDs, and other extra fields.")
    result = frame.loc[:, features].copy()
    ranges = {"G1": (0, 20), "G2": (0, 20), "studytime": (1, 4), "absences": (0, 93), "failures": (0, 4), "traveltime": (1, 4)}
    for col in result.columns:
        if col in ranges:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            if result[col].isna().any():
                raise ValueError(f"{col} must contain numeric values.")
            low, high = ranges[col]
            if not result[col].between(low, high).all():
                raise ValueError(f"{col} values must be between {low} and {high}.")
            if col in {"studytime", "failures", "traveltime"} and not result[col].mod(1).eq(0).all():
                raise ValueError(f"{col} values must be whole-number categories.")
            if col == "failures" and not result[col].isin([0, 1, 2, 4]).all():
                raise ValueError("failures must use the dataset categories 0, 1, 2, or 4 (4 represents three or more).")
        elif result[col].isna().any():
            raise ValueError(f"{col} contains missing values.")
        else:
            result[col] = result[col].astype(str).str.lower()
            if not result[col].isin(["yes", "no"]).all():
                raise ValueError(f"{col} values must be 'yes' or 'no'.")
    if result.empty:
        raise ValueError("Upload must contain at least one student row.")
    return result
