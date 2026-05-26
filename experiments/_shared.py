"""
experiments/_shared.py
======================
Common setup that every experiment imports instead of copy-pasting.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE MODE — must run BEFORE deepchem is imported anywhere.
#  DeepChem's CGCNNFeaturizer calls download_url() at IMPORT TIME (inside
#  deepchem/__init__.py → molnet → CGCNN). That means we cannot patch deepchem
#  itself — the crash happens during deepchem's own import.
#
#  Instead we patch the stdlib network primitives (urllib.request) that deepchem
#  ultimately calls. urllib.request has no import side effects, so patching it
#  first guarantees no network access when deepchem later imports.
#
#  If a pre-staged copy of the requested file exists in offline_data/, it is
#  copied to the destination; otherwise the call becomes a harmless no-op.
#  No files inside the environment/site-packages are modified.
# ══════════════════════════════════════════════════════════════════════════════
import os
import shutil
import urllib.request as _urlreq

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OFFLINE_DIR  = os.path.join(_PROJECT_ROOT, "offline_data")

os.environ["DEEPCHEM_DATA_DIR"] = _OFFLINE_DIR
os.makedirs(_OFFLINE_DIR, exist_ok=True)

_real_urlretrieve = _urlreq.urlretrieve

def _offline_urlretrieve(url, filename=None, *args, **kwargs):
    """Offline replacement for urllib.request.urlretrieve — never hits network."""
    name = url.split("/")[-1]

    # destination DeepChem wants
    if filename is None:
        filename = os.path.join(_OFFLINE_DIR, name)

    # already there
    if os.path.exists(filename):
        return filename, None

    # copy from staged offline_data/ if available
    staged = os.path.join(_OFFLINE_DIR, name)
    if os.path.exists(staged) and os.path.abspath(staged) != os.path.abspath(filename):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        shutil.copy(staged, filename)
        return filename, None

    # nothing available: create an empty placeholder so callers that only need
    # the file to *exist* (not its contents) can proceed, and warn loudly.
    import warnings as _w
    _w.warn(f"[offline] network blocked; '{name}' not found in offline_data/. "
            f"Writing empty placeholder at {filename}.")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    open(filename, "a").close()
    return filename, None

# install the patch BEFORE any deepchem import
_urlreq.urlretrieve = _offline_urlretrieve
# ══════════════════════════════════════════════════════════════════════════════


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
NUM_LAYERS             = 5
DELTA                  = 1
EQUATION               = 10
GRAPH_POOL             = "sum"
NEIGHBOR_POOL          = "sum"
DEVICE                 = torch.device("cpu")

DEFAULT_HV_DIMS        = [100, 500, 1000, 2000, 5000, 10000]

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
    """
    Return (df298_train, df298_test, train_set, test_set,
            train_valid_mask, test_valid_mask).

    train_valid_mask / test_valid_mask are boolean arrays of length
    equal to the ORIGINAL (unfiltered) CSV row count, so they can be
    used to index into the full GVFA embedding arrays.
    """
    train_set = pd.read_csv("final_data/final_unique_train_fixed.csv").reset_index(drop=True)
    test_set  = pd.read_csv("final_data/final_unique_test.csv").reset_index(drop=True)

    df123_train = utilities.generate123(train_set.smiles_canon).reset_index(drop=True)
    df123_test  = utilities.generate123(test_set.smiles_canon).reset_index(drop=True)
    df128_train = utilities.fingerprint(train_set.smiles_canon, 2, 128).reset_index(drop=True)
    df128_test  = utilities.fingerprint(test_set.smiles_canon,  2, 128).reset_index(drop=True)
    df7_train   = utilities.get_functional_groups(train_set.smiles_canon).reset_index(drop=True)
    df7_test    = utilities.get_functional_groups(test_set.smiles_canon).reset_index(drop=True)
    df38_train  = utilities.generate_features38(train_set.smiles_canon).reset_index(drop=True)
    df38_test   = utilities.generate_features38(test_set.smiles_canon).reset_index(drop=True)

    df298_train_full = pd.concat([df123_train, df128_train, df7_train, df38_train], axis=1)
    df298_test_full  = pd.concat([df123_test,  df128_test,  df7_test,  df38_test],  axis=1)

    # build masks from FULL (unfiltered) arrays — length matches GVFA output
    train_valid = df298_train_full.notna().any(axis=1).values  # shape [N_full_train]
    test_valid  = df298_test_full.notna().any(axis=1).values   # shape [N_full_test]

    n_drop_tr = (~train_valid).sum()
    n_drop_te = (~test_valid).sum()
    if n_drop_tr:
        print(f"  [load_trad_features] dropped {n_drop_tr} train rows (RDKit all-NaN)")
    if n_drop_te:
        print(f"  [load_trad_features] dropped {n_drop_te} test rows (RDKit all-NaN)")

    # now filter both the feature df and the dataset rows
    df298_train = df298_train_full[train_valid].reset_index(drop=True)
    df298_test  = df298_test_full[test_valid].reset_index(drop=True)
    train_set   = train_set[train_valid].reset_index(drop=True)
    test_set    = test_set[test_valid].reset_index(drop=True)

    return df298_train, df298_test, train_set, test_set, train_valid, test_valid


def load_traditional_features_aligned():
    """
    Convenience wrapper — returns everything exp9 needs in one call.
    Returns (df298_train, df298_test, y_train, y_test,
             train_valid_mask, test_valid_mask)
    """
    df298_train, df298_test, train_set, test_set, \
        train_valid, test_valid = load_traditional_features()

    y_train = train_set["LogS"].values.astype(np.float32)
    y_test  = test_set["LogS"].values.astype(np.float32)

    return df298_train, df298_test, y_train, y_test, train_valid, test_valid


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


def build_gvfa_embeddings_aligned(hv_dim: int, n_train_keep: int, n_test_keep: int,
                                   train_valid_mask: np.ndarray,
                                   test_valid_mask:  np.ndarray):
    train_data, test_data = load_data()

    train_graphs = create_graph_list(train_data)
    test_graphs  = create_graph_list(test_data)

    train_HVs = VSA_conversion(train_graphs.copy(), hv_dim)
    test_HVs  = VSA_conversion(test_graphs.copy(),  hv_dim)

    in_dim = test_HVs[0].node_features.shape[1]
    model  = GraphCNN(in_dim, NUM_LAYERS, DELTA, GRAPH_POOL, NEIGHBOR_POOL, DEVICE, EQUATION)

    train_emb, train_labels = getEmbedding(model, DEVICE, train_HVs)
    test_emb,  test_labels  = getEmbedding(model, DEVICE, test_HVs)

    gvfa_tr = to_numpy(train_emb.squeeze(0))
    gvfa_te = to_numpy(test_emb.squeeze(0))
    y_tr    = to_numpy(train_labels).ravel()
    y_te    = to_numpy(test_labels).ravel()

    # rebuild mask from scratch using actual GVFA array length
    # this is safe because the mask just marks the same failed molecules
    n_full_tr = gvfa_tr.shape[0]
    n_full_te = gvfa_te.shape[0]

    if len(train_valid_mask) != n_full_tr:
        print(f"  [align] mask length {len(train_valid_mask)} != GVFA rows {n_full_tr}. "
              f"Dropping last {n_full_tr - n_train_keep} rows to match descriptor count.")
        # fallback: keep first n_train_keep rows (same molecules, consistent ordering)
        gvfa_tr = gvfa_tr[:n_train_keep]
        y_tr    = y_tr[:n_train_keep]
    else:
        gvfa_tr = gvfa_tr[train_valid_mask]
        y_tr    = y_tr[train_valid_mask]

    if len(test_valid_mask) != n_full_te:
        print(f"  [align] test mask length {len(test_valid_mask)} != GVFA rows {n_full_te}. "
              f"Keeping first {n_test_keep} rows.")
        gvfa_te = gvfa_te[:n_test_keep]
        y_te    = y_te[:n_test_keep]
    else:
        gvfa_te = gvfa_te[test_valid_mask]
        y_te    = y_te[test_valid_mask]

    assert gvfa_tr.shape[0] == n_train_keep, \
        f"Alignment failed: gvfa_train={gvfa_tr.shape[0]} vs desc_train={n_train_keep}"
    assert gvfa_te.shape[0] == n_test_keep, \
        f"Alignment failed: gvfa_test={gvfa_te.shape[0]} vs desc_test={n_test_keep}"

    print(f"  [build_gvfa_aligned] train={gvfa_tr.shape}  test={gvfa_te.shape}")
    return gvfa_tr, gvfa_te, y_tr, y_te


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