import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import DATA_PATH, EARLY_FEATURES, LATE_FEATURES, TARGET, load_data
from src.evaluation import regression_metrics
from src.preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SEED = 42


def build_pipeline(frame: pd.DataFrame, features: list[str], estimator):
    categorical_names = {"schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet"}
    categorical = [name for name in features if name in categorical_names]
    numeric = [name for name in features if name not in categorical]
    return Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", estimator)])


def train_experiment(frame: pd.DataFrame, features: list[str], stage: str):
    train_x, test_x, train_y, test_y = train_test_split(
        frame[features], frame[TARGET], test_size=0.2, random_state=SEED
    )
    estimators = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1),
    }
    metrics = {}
    fitted = {}
    for name, estimator in estimators.items():
        pipeline = build_pipeline(frame, features, estimator)
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(test_x)
        fitted[name] = pipeline
        metrics[name] = regression_metrics(test_y, prediction)
    cv_mae = {}
    for name in ("LinearRegression", "RandomForestRegressor"):
        cv_pipeline = build_pipeline(frame, features, estimators[name])
        scores = cross_val_score(cv_pipeline, train_x, train_y, cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
        cv_mae[name] = float(-scores.mean())
    selected_name = min(cv_mae, key=cv_mae.get)
    output = pd.DataFrame({"actual": test_y.to_numpy(), "predicted": fitted[selected_name].predict(test_x)})
    joblib.dump(fitted[selected_name], MODEL_DIR / f"{stage}.joblib")
    output.to_csv(MODEL_DIR / f"{stage}-holdout.csv", index=False)
    return {"selected_model": selected_name, "selection_cv_mae": cv_mae, "features": features, "rows_train": len(train_x), "rows_test": len(test_x), "metrics": metrics}


def train_all(data_path: str | Path = DATA_PATH) -> dict:
    frame = load_data(data_path)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": "UCI Student Performance (Mathematics)",
        "rows": len(frame),
        "split": "80/20 random train/test; random_state=42",
        "target": TARGET,
        "early": train_experiment(frame, EARLY_FEATURES, "early"),
        "late": train_experiment(frame, LATE_FEATURES, "late"),
    }
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(train_all(), indent=2))
