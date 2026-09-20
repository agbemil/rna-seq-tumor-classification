import pandas as pd
from src.data import audit_dataset


def test_audit_dataset():
    X = pd.DataFrame({"a": [1, 2], "b": [0, 0]})
    y = pd.Series(["A", "B"])
    result = audit_dataset(X, y)
    assert result["samples"] == 2
    assert result["features"] == 2
    assert result["constant_features"] == 1
