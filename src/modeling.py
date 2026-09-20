"""Leakage-free high-dimensional classification pipelines."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_model_specs(random_state=42):
    return {
        "Logistic Regression": (
            Pipeline([
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                )),
            ]),
            {
                "select__k": [100, 250, 500],
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "Linear SVM": (
            Pipeline([
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif)),
                ("scale", StandardScaler()),
                ("model", SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                )),
            ]),
            {
                "select__k": [100, 250, 500],
                "model__C": [0.01, 0.1, 1.0],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(f_classif)),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]),
            {"select__k": [250, 500, 1000]},
        ),
    }


def tune_models(X_train, X_test, y_train, y_test, random_state=42):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    rows = []
    fitted = {}

    for name, (pipeline, grid) in get_model_specs(random_state).items():
        start = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        pred = search.predict(X_test)
        proba = search.predict_proba(X_test)

        rows.append({
            "Model": name,
            "Best CV Macro F1": search.best_score_,
            "Test Accuracy": accuracy_score(y_test, pred),
            "Test Macro F1": f1_score(y_test, pred, average="macro"),
            "Test Weighted F1": f1_score(
                y_test, pred, average="weighted"
            ),
            "Test Macro ROC-AUC (OvR)": roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",
                average="macro",
            ),
            "Runtime Seconds": time.perf_counter() - start,
            "Best Parameters": json.dumps(search.best_params_),
        })
        fitted[name] = search

    return pd.DataFrame(rows), fitted


def selected_feature_table(best_estimator, original_columns):
    variance = best_estimator.named_steps["variance"]
    selector = best_estimator.named_steps["select"]

    nonconstant = np.asarray(original_columns)[variance.get_support()]
    mask = selector.get_support()

    return pd.DataFrame({
        "Feature": nonconstant[mask],
        "ANOVA_F_Score": selector.scores_[mask],
        "P_Value": selector.pvalues_[mask],
    }).sort_values("ANOVA_F_Score", ascending=False)
