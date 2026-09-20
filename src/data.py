"""Load and audit the UCI RNA-Seq tumor classification dataset."""

from pathlib import Path
import warnings

import pandas as pd

EXPECTED_UCI_FEATURES = 20531


def load_local_data(data_dir: Path):
    data_path = data_dir / "data.csv"
    labels_path = data_dir / "labels.csv"

    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Place official data.csv and labels.csv in data/, or use --source uci."
        )

    X = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)

    X = X.loc[:, ~X.columns.str.startswith("Unnamed:")]

    if "Class" not in labels.columns:
        raise ValueError("labels.csv must contain a 'Class' column.")

    y = labels["Class"].astype(str)

    if len(X) != len(y):
        raise ValueError("Feature and label files have different row counts.")

    if X.shape[1] != EXPECTED_UCI_FEATURES:
        warnings.warn(
            f"Expected {EXPECTED_UCI_FEATURES} UCI features; "
            f"found {X.shape[1]}. The local copy may be incomplete.",
            stacklevel=2,
        )

    return X, y


def fetch_uci_data():
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=401)
    X = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if isinstance(targets, pd.DataFrame):
        y = (
            targets["Class"].astype(str)
            if "Class" in targets.columns
            else targets.iloc[:, 0].astype(str)
        )
    else:
        y = pd.Series(targets).astype(str)

    X = X.loc[:, ~X.columns.str.startswith("Unnamed:")]
    return X, y


def audit_dataset(X, y):
    return {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "constant_features": int((X.nunique(dropna=False) <= 1).sum()),
        "class_counts": y.value_counts().sort_index().to_dict(),
    }
