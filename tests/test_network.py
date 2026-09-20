import pandas as pd
from src.network import build_coexpression_network


def test_correlated_features_are_connected():
    X = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": [4, 1, 3, 2],
    })
    graph = build_coexpression_network(
        X,
        ["a", "b", "c"],
        threshold=0.9,
    )
    assert graph.has_edge("a", "b")
