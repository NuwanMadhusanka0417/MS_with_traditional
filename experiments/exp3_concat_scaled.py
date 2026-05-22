"""
Exp 3 – Concatenation: 298 descriptors (StandardScaler) + GVFA.
Fixed HV dim = 2000.  Regressor: XGBoost.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_traditional_features,
    build_gvfa_embeddings,
    fit_xgb_and_report,
    to_numpy,
    clean_nan,
)

HV_DIM = 2000


def run():
    print(f"\n[Exp 3] 298 (StandardScaler) + GVFA(D={HV_DIM})")

    df298_train, df298_test, _, _ = load_traditional_features()

    scaler = StandardScaler()
    df298_train_scaled = scaler.fit_transform(df298_train.values)
    df298_test_scaled  = scaler.transform(df298_test.values)

    df_torch_train = torch.from_numpy(df298_train_scaled.astype(np.float32))
    df_torch_test  = torch.from_numpy(df298_test_scaled.astype(np.float32))

    train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(HV_DIM)

    X_train = torch.cat([df_torch_train, train_emb], dim=1)
    X_test  = torch.cat([df_torch_test,  test_emb],  dim=1)

    X_train_np, X_test_np, y_train, y_test = to_numpy(
        X_train, X_test, train_labels, test_labels
    )
    X_train_np = clean_nan(X_train_np)
    X_test_np  = clean_nan(X_test_np)

    return fit_xgb_and_report(
        X_train_np, y_train.ravel(),
        X_test_np,  y_test.ravel(),
        label=f"XGB_298(StdScaler)+GVFA(D={HV_DIM})",
    )
