import pandas as pd
import pytest

from src.data_loader import DATA_PATH, EARLY_FEATURES, LATE_FEATURES, TARGET, load_data


def test_dataset_loads_and_has_expected_columns():
    data = load_data()
    assert len(data) == 395
    assert set(EARLY_FEATURES + LATE_FEATURES + [TARGET]).issubset(data.columns)


def test_missing_data_file_is_reported():
    with pytest.raises(FileNotFoundError):
        load_data(DATA_PATH.parent / "this-file-does-not-exist.csv")
