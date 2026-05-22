"""
Exp 6 – Separate StandardScaler applied independently to the 298-descriptor
block and to the GVFA embedding block before concatenation.
Regressor: XGBoost across all HV dims.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

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

    results = []

    for dim in hv_dims:
        print(f"\n[Exp 6] HV dim = {dim} | separate scalers per block")

        train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(dim)

        # scale each block independently
        sc_desc = StandardScaler()
        df_tr_s = sc_desc.fit_transform(df298_train.values.astype(np.float32))
        df_te_s = sc_desc.transform(df298_test.values.astype(np.float32))

        sc_hv = StandardScaler()
        hv_tr_np, hv_te_np = to_numpy(train_emb, test_emb)
        hv_tr_s = sc_hv.fit_transform(hv_tr_np)
        hv_te_s = sc_hv.transform(hv_te_np)

        X_train_np = np.concatenate([df_tr_s, hv_tr_s], axis=1)
        X_test_np  = np.concatenate([df_te_s, hv_te_s], axis=1)

        y_train, y_test = to_numpy(train_labels, test_labels)
        X_train_np = clean_nan(X_train_np)
        X_test_np  = clean_nan(X_test_np)

        m = fit_xgb_and_report(
            X_train_np, y_train.ravel(),
            X_test_np,  y_test.ravel(),
            label=f"XGB_298(sep-scale)+GVFA(D={dim})",
        )
        results.append(m)

    return results
