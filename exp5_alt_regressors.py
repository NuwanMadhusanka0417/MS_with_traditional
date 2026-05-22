"""
Exp 5 – Alternative regressors: SVR-RBF / SVR-Linear / Ridge.
Uses 298 descriptors (StandardScaler) + GVFA at HV dim=2000.
A second StandardScaler is applied to the full concatenated feature space
(recommended for SVR and Ridge).
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from experiments._shared import (
    DESC_DETAIL,
    load_traditional_features,
    build_gvfa_embeddings,
    to_numpy,
    clean_nan,
)
from src import utilities

HV_DIM = 2000


def run():
    print(f"\n[Exp 5] Alt regressors (SVR-RBF / SVR-Lin / Ridge) — D={HV_DIM}")

    df298_train, df298_test, _, _ = load_traditional_features()

    scaler_desc = StandardScaler()
    df_torch_train = torch.from_numpy(
        scaler_desc.fit_transform(df298_train.values).astype(np.float32)
    )
    df_torch_test = torch.from_numpy(
        scaler_desc.transform(df298_test.values).astype(np.float32)
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

    # second scaler for linear/SVM models
    scaler_full = StandardScaler()
    X_train_s   = scaler_full.fit_transform(X_train_np)
    X_test_s    = scaler_full.transform(X_test_np)

    results = {}

    # ── SVR RBF ──────────────────────────────────────────────────────────────
    print("  Fitting SVR-RBF … (may take a few minutes on large datasets)")
    svr_rbf = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    svr_rbf.fit(X_train_s, y_train)
    preds = svr_rbf.predict(X_test_s)
    m = utilities.get_errors1(y_test, preds, f"SVR-RBF_298+GVFA(D={HV_DIM})")
    m["Descriptors_Detail"] = DESC_DETAIL
    print("  SVR RBF:", m)
    results["svr_rbf"] = m

    # ── SVR Linear ───────────────────────────────────────────────────────────
    print("  Fitting SVR-Linear …")
    svr_lin = SVR(kernel="linear", C=1.0, epsilon=0.1)
    svr_lin.fit(X_train_s, y_train)
    preds = svr_lin.predict(X_test_s)
    m = utilities.get_errors1(y_test, preds, f"SVR-Lin_298+GVFA(D={HV_DIM})")
    m["Descriptors_Detail"] = DESC_DETAIL
    print("  SVR Linear:", m)
    results["svr_lin"] = m

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print("  Fitting Ridge …")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_s, y_train)
    preds = ridge.predict(X_test_s)
    m = utilities.get_errors1(y_test, preds, f"Ridge_298+GVFA(D={HV_DIM})")
    m["Descriptors_Detail"] = DESC_DETAIL
    print("  Ridge:", m)
    results["ridge"] = m

    return results
