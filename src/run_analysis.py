"""Run the complete RNA-Seq tumor classification workflow."""

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data import audit_dataset, fetch_uci_data, load_local_data
from .modeling import selected_feature_table, tune_models
from .network import build_coexpression_network, community_summary


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["local", "uci"],
        default="local",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.source == "uci":
        X, y = fetch_uci_data()
    else:
        X, y = load_local_data(args.data_dir)

    (RESULTS / "dataset_summary.json").write_text(
        json.dumps(audit_dataset(X, y), indent=2),
        encoding="utf-8",
    )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        stratify=y_encoded,
        random_state=42,
    )

    comparison, fitted = tune_models(
        X_train, X_test, y_train, y_test
    )
    comparison.to_csv(
        RESULTS / "model_comparison.csv",
        index=False,
    )

    flagship = fitted["Logistic Regression"]
    pred = flagship.predict(X_test)

    report = pd.DataFrame(
        classification_report(
            y_test,
            pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(
        RESULTS / "classification_report_logistic.csv"
    )

    cm = confusion_matrix(y_test, pred)
    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_,
    ).to_csv(
        RESULTS / "confusion_matrix_logistic.csv"
    )

    selected = selected_feature_table(
        flagship.best_estimator_,
        X.columns,
    )
    selected.to_csv(
        RESULTS / "selected_features_logistic.csv",
        index=False,
    )

    top50 = selected.head(50)["Feature"].tolist()
    graph = build_coexpression_network(
        X_train,
        top50,
        threshold=0.70,
    )
    communities, summary, null_scores = community_summary(graph)

    communities.to_csv(
        RESULTS / "coexpression_communities.csv",
        index=False,
    )
    nx.write_graphml(
        graph,
        RESULTS / "coexpression_network.graphml",
    )
    (RESULTS / "coexpression_network_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
