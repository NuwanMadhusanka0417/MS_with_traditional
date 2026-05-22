"""
Exp 4 – XGBoost n_estimators sweep (500 / 1000 / 2000 / 5000).
Uses 298 descriptors (StandardScaler) + GVFA at HV dim=2000.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from experiments._shared import (
    XGB_PARAMS,
    DESC_DETAIL,
    load_traditional_features,
    build_gvfa_embeddings,
    to_numpy,
    clean_nan,
)
from src import utilities

HV_DIM        = 2000
N_EST_SWEEP   = [500, 1000, 2000, 5000]


def run():
    print(f"\n[Exp 4] XGB n_estimators sweep — 298 (StdScaler) + GVFA(D={HV_DIM})")

    df298_train, df298_test, _, _ = load_traditional_features()

    scaler = StandardScaler()
    df_torch_train = torch.from_numpy(
        scaler.fit_transform(df298_train.values).astype(np.float32)
    )
    df_torch_test = torch.from_numpy(
        scaler.transform(df298_test.values).astype(np.float32)
    )

    train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(HV_DIM)

    X_train = torch.cat([df_torch_train, train_emb], dim=1)
    X_test  = torch.cat([df_torch_test,  test_emb],  dim=1)

    X_train_np, X_test_np, y_train, y_test = to_numpy(
        X_train, X_test, train_labels, test_labels
    )
    X_train_np = clean_nan(X_train_np)
    X_test_np  = clean_nan(X_test_np)
    y_train    = y_train.ravel()
    y_test     = y_test.ravel()

    results = []
    for n_est in N_EST_SWEEP:
        params = {**XGB_PARAMS, "n_estimators": n_est}
        xgb = XGBRegressor(**params)
        xgb.fit(X_train_np, y_train,
                eval_set=[(X_test_np, y_test)],
                verbose=False)
        preds   = xgb.predict(X_test_np)
        metrics = utilities.get_errors1(
            y_test, preds, f"XGB(n_est={n_est})_298+GVFA(D={HV_DIM})"
        )
        metrics["Descriptors_Detail"] = DESC_DETAIL
        print(n_est, metrics)
        results.append(metrics)

    return results
