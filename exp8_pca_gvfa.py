"""
Exp 8 – PCA compression of GVFA embeddings → 100 components, then
concatenate with raw 298-dim traditional descriptors.
Regressor: XGBoost across selected HV dims.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA

from experiments._shared import (
    load_traditional_features,
    build_gvfa_embeddings,
    fit_xgb_and_report,
    to_numpy,
    clean_nan,
)

HV_DIMS  = [1000, 2000, 5000, 10000]
PCA_DIMS = 100


def run(hv_dims=None):
    hv_dims = hv_dims or HV_DIMS

    df298_train, df298_test, _, _ = load_traditional_features()

    df_torch_train = torch.from_numpy(df298_train.to_numpy(dtype="float32"))
    df_torch_test  = torch.from_numpy(df298_test.to_numpy(dtype="float32"))

    results = []

    for dim in hv_dims:
        print(f"\n[Exp 8] HV dim = {dim} | PCA({PCA_DIMS}) on GVFA, then concat 298(raw)")

        train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(dim)

        gvfa_tr, gvfa_te = to_numpy(train_emb, test_emb)

        pca = PCA(n_components=PCA_DIMS, random_state=42)
        gvfa_tr_pca = pca.fit_transform(gvfa_tr).astype(np.float32)
        gvfa_te_pca = pca.transform(gvfa_te).astype(np.float32)

        gvfa_tr_t = torch.from_numpy(gvfa_tr_pca)
        gvfa_te_t = torch.from_numpy(gvfa_te_pca)

        X_train = torch.cat([df_torch_train, gvfa_tr_t], dim=1)
        X_test  = torch.cat([df_torch_test,  gvfa_te_t], dim=1)

        X_train_np, X_test_np, y_train, y_test = to_numpy(
            X_train, X_test, train_labels, test_labels
        )
        X_train_np = clean_nan(X_train_np)
        X_test_np  = clean_nan(X_test_np)

        m = fit_xgb_and_report(
            X_train_np, y_train.ravel(),
            X_test_np,  y_test.ravel(),
            label=f"XGB_298(raw)+PCA{PCA_DIMS}(GVFA D={dim})",
        )
        results.append(m)

    return results
