"""
Exp 7 – Superposition / Bundling:
  1. Project 298-dim descriptors → HV_dim via GaussianRandomProjection.
  2. Add (superpose) the projected descriptor HV and the GVFA HV element-wise.
  3. L2-normalise the summed HV per molecule.
Regressor: XGBoost across all HV dims.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from experiments._shared import (
    DEFAULT_HV_DIMS,
    load_traditional_features,
    build_gvfa_embeddings,
    fit_xgb_and_report,
    to_numpy,
)


def _make_rp_pipeline(rp_dim: int) -> Pipeline:
    """RobustScaler → GaussianRandomProjection pipeline for the 298-desc block."""
    return Pipeline([
        ("impute",  SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler(with_centering=True, with_scaling=True)),
        ("rp",      GaussianRandomProjection(n_components=rp_dim, random_state=42)),
    ])


def _l2_norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def run(hv_dims=None):
    hv_dims = hv_dims or DEFAULT_HV_DIMS

    df298_train, df298_test, _, _ = load_traditional_features()

    # replace infs with nan so the pipeline can handle them
    df_tr = df298_train.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    df_te = df298_test.replace([np.inf,  -np.inf], np.nan).astype(np.float32)

    results = []

    for dim in hv_dims:
        print(f"\n[Exp 7] HV dim = {dim} | bundling (RP desc + GVFA, element-wise add)")

        train_emb, test_emb, train_labels, test_labels = build_gvfa_embeddings(dim)

        pipe = _make_rp_pipeline(dim)
        desc_tr_rp = pipe.fit_transform(df_tr).astype(np.float32)
        desc_te_rp = pipe.transform(df_te).astype(np.float32)

        gvfa_tr, gvfa_te = to_numpy(train_emb, test_emb)

        # superpose (add)
        S_train = _l2_norm(desc_tr_rp + gvfa_tr)
        S_test  = _l2_norm(desc_te_rp + gvfa_te)

        y_train, y_test = to_numpy(train_labels, test_labels)

        m = fit_xgb_and_report(
            S_train, y_train.ravel(),
            S_test,  y_test.ravel(),
            label=f"XGB_298(bundled)+GVFA(D={dim})",
        )
        results.append(m)

    return results
