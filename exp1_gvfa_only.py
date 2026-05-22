"""
Exp 1 – GVFA-only embeddings (no traditional descriptors) across HV dims.
Regressor: XGBoost
"""

import numpy as np
from experiments._shared import (
    DEFAULT_HV_DIMS,
    build_gvfa_embeddings,
    fit_xgb_and_report,
    to_numpy,
    clean_nan,
)


def run(hv_dims=None):
    hv_dims = hv_dims or DEFAULT_HV_DIMS
    results = []

    for dim in hv_dims:
        print(f"\n[Exp 1] HV dim = {dim}")
        train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(dim)

        X_train, X_test, y_train, y_test = to_numpy(
            train_emb, test_emb, train_labels, test_labels
        )
        X_train = clean_nan(X_train)
        X_test  = clean_nan(X_test)

        m = fit_xgb_and_report(
            X_train, y_train.ravel(),
            X_test,  y_test.ravel(),
            label=f"GVFA-only (D={dim})",
        )
        results.append(m)

    return results
