"""Correlation-based co-expression network analysis."""

import numpy as np
import pandas as pd
import networkx as nx


def build_coexpression_network(X_train, feature_names, threshold=0.70):
    corr = X_train[feature_names].corr().abs()

    graph = nx.Graph()
    graph.add_nodes_from(feature_names)

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            value = float(corr.iloc[i, j])
            if value >= threshold:
                graph.add_edge(
                    feature_names[i],
                    feature_names[j],
                    weight=value,
                )

    return graph


def community_summary(graph, permutations=200):
    communities = list(
        nx.algorithms.community.greedy_modularity_communities(graph)
    )
    modularity = nx.algorithms.community.modularity(
        graph, communities
    )

    rows = []
    for community_id, members in enumerate(communities, start=1):
        for feature in sorted(members):
            rows.append({
                "Community": community_id,
                "Feature": feature,
                "Degree": graph.degree(feature),
            })

    base = nx.Graph()
    base.add_nodes_from(graph.nodes())
    base.add_edges_from(graph.edges())

    null_scores = []
    for b in range(permutations):
        randomized = base.copy()
        nx.double_edge_swap(
            randomized,
            nswap=5 * randomized.number_of_edges(),
            max_tries=100 * randomized.number_of_edges(),
            seed=1000 + b,
        )
        null_communities = list(
            nx.algorithms.community.greedy_modularity_communities(
                randomized
            )
        )
        null_scores.append(
            nx.algorithms.community.modularity(
                randomized,
                null_communities,
            )
        )

    null_scores = np.asarray(null_scores)
    p_value = (
        1 + np.sum(null_scores >= modularity)
    ) / (1 + len(null_scores))

    summary = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
        "communities": len(communities),
        "modularity": float(modularity),
        "randomizations": int(len(null_scores)),
        "null_mean_modularity": float(null_scores.mean()),
        "permutation_p_value": float(p_value),
    }

    return pd.DataFrame(rows), summary, null_scores
