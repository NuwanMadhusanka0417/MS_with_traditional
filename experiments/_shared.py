"""
experiments/_shared.py
======================
Common setup that every experiment imports instead of copy-pasting.
"""

import warnings
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

# ── silence noisy logs ────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── project imports ───────────────────────────────────────────────────────────
from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
from src import utilities

# ── constants shared across all experiments ───────────────────────────────────
NUM_LAYERS             = 4
DELTA                  = 0
EQUATION               = 10
GRAPH_POOL             = "sum"
NEIGHBOR_POOL          = "sum"
DEVICE                 = torch.device("cpu")

DEFAULT_HV_DIMS        = [1000, 2000, 5000, 10000]

XGB_PARAMS = dict(
    n_estimators    = 2000,
    learning_rate   = 0.03,
    max_depth       = 7,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_lambda      = 1.0,
    reg_alpha       = 0.0,
    random_state    = 42,
    n_jobs          = 4,
    tree_method     = "hist",
)

DESC_DETAIL = "123 RDKit + 128 Morgan FP + 7 func-groups + 38 engineered"


# ── reusable building blocks ──────────────────────────────────────────────────

def load_traditional_features():
    """Return (df298_train, df298_test, train_set, test_set)."""
    train_set = pd.read_csv("final_data/final_unique_train_fixed.csv")
    test_set  = pd.read_csv("final_data/final_unique_test.csv")

    df123_train = utilities.generate123(train_set.smiles_canon)
    df123_test  = utilities.generate123(test_set.smiles_canon)
    df128_train = utilities.fingerprint(train_set.smiles_canon, 2, 128)
    df128_test  = utilities.fingerprint(test_set.smiles_canon,  2, 128)
    df7_train   = utilities.get_functional_groups(train_set.smiles_canon)
    df7_test    = utilities.get_functional_groups(test_set.smiles_canon)
    df38_train  = utilities.generate_features38(train_set.smiles_canon)
    df38_test   = utilities.generate_features38(test_set.smiles_canon)

    df298_train = pd.concat([df123_train, df128_train, df7_train, df38_train], axis=1)
    df298_test  = pd.concat([df123_test,  df128_test,  df7_test,  df38_test],  axis=1)

    return df298_train, df298_test, train_set, test_set


def build_gvfa_embeddings(hv_dim: int):
    """
    Load graph data, run VSA conversion and GraphCNN forward pass.
    Returns (train_emb, test_emb, train_labels, test_labels) — all torch tensors,
    embeddings are [N, D].
    """
    train_data, test_data = load_data()

    train_graphs = create_graph_list(train_data)
    test_graphs  = create_graph_list(test_data)

    train_HVs = VSA_conversion(train_graphs.copy(), hv_dim)
    test_HVs  = VSA_conversion(test_graphs.copy(),  hv_dim)

    in_dim = test_HVs[0].node_features.shape[1]
    model  = GraphCNN(in_dim, NUM_LAYERS, DELTA, GRAPH_POOL, NEIGHBOR_POOL, DEVICE, EQUATION)

    train_emb, train_labels = getEmbedding(model, DEVICE, train_HVs)
    test_emb,  test_labels  = getEmbedding(model, DEVICE, test_HVs)

    return (
        train_emb.squeeze(0),
        test_emb.squeeze(0),
        train_labels,
        test_labels,
    )


def fit_xgb_and_report(X_train, y_train, X_test, y_test, label: str):
    """Train XGBoost, print metrics, return metrics dict."""
    from xgboost import XGBRegressor
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    preds   = xgb.predict(X_test)
    metrics = utilities.get_errors1(y_test, preds, label)
    metrics["Descriptors_Detail"] = DESC_DETAIL
    print(metrics)
    return metrics


def to_numpy(*tensors):
    """Convert torch tensors or numpy arrays to float32 numpy."""
    out = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            out.append(t.detach().cpu().numpy().astype(np.float32))
        else:
            out.append(np.asarray(t, dtype=np.float32))
    return out if len(out) > 1 else out[0]


def clean_nan(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
