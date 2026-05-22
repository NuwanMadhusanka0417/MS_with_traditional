"""
Exp 2 – Concatenation: 298 traditional descriptors (raw, no scaling) + GVFA.
Regressor: XGBoost
Matches the first multi-dim loop in the notebook.
"""

import numpy as np
import torch
from experiments._shared import (
    DEFAULT_HV_DIMS,
    load_traditional_features,
    build_gvfa_embeddings,
    fit_xgb_and_report,
    to_numpy,
    clean_nan,
)


def run(hv_dims=None):
    hv_dims = hv_dims or DEFAULT_HV_DIMS

    df298_train, df298_test, _, _ = load_traditional_features()

    df_torch_train = torch.from_numpy(df298_train.to_numpy(dtype="float32"))
    df_torch_test  = torch.from_numpy(df298_test.to_numpy(dtype="float32"))

    results = []

    for dim in hv_dims:
        print(f"\n[Exp 2] HV dim = {dim} | 298 raw + GVFA concat")
        train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(dim)

        X_train = torch.cat([df_torch_train, train_emb], dim=1)
        X_test  = torch.cat([df_torch_test,  test_emb],  dim=1)

        X_train_np, X_test_np, y_train, y_test = to_numpy(
            X_train, X_test, train_labels, test_labels
        )
        X_train_np = clean_nan(X_train_np)
        X_test_np  = clean_nan(X_test_np)

        m = fit_xgb_and_report(
            X_train_np, y_train.ravel(),
            X_test_np,  y_test.ravel(),
            label=f"XGB_298(raw)+GVFA(D={dim})",
        )
        results.append(m)

    return results
