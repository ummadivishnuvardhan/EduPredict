import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.data_loader import EARLY_FEATURES, load_data
from src.predict import predict_grades, risk_category, validate_batch
from src.train import build_pipeline


def test_training_and_inference_are_bounded():
    data = load_data()
    model = build_pipeline(data, EARLY_FEATURES, DummyRegressor(strategy="mean"))
    model.fit(data[EARLY_FEATURES], data.G3)
    predicted = predict_grades(model, data[EARLY_FEATURES].head(5))
    assert len(predicted) == 5
    assert np.all((predicted >= 0) & (predicted <= 20))


def test_risk_threshold():
    assert risk_category(9.9, 10) == "Review recommended"
    assert risk_category(10, 10) == "At/above illustrative threshold"
    with pytest.raises(ValueError):
        risk_category(9, 21)


def test_batch_validation():
    data = load_data()
    clean = validate_batch(data[EARLY_FEATURES].head(2), EARLY_FEATURES)
    assert list(clean.columns) == EARLY_FEATURES
    with pytest.raises(ValueError, match="Missing required"):
        validate_batch(pd.DataFrame({"G1": [5]}), EARLY_FEATURES)
    identified = data[EARLY_FEATURES].head(1).assign(student_id=["example"])
    with pytest.raises(ValueError, match="Remove names, IDs"):
        validate_batch(identified, EARLY_FEATURES)
    invalid = data[EARLY_FEATURES].head(1).copy()
    invalid.loc[invalid.index[0], "G1"] = 25
    with pytest.raises(ValueError, match="between 0 and 20"):
        validate_batch(invalid, EARLY_FEATURES)
    invalid_category = data[EARLY_FEATURES].head(1).copy()
    invalid_category.loc[invalid_category.index[0], "failures"] = 3
    with pytest.raises(ValueError, match="dataset categories"):
        validate_batch(invalid_category, EARLY_FEATURES)
