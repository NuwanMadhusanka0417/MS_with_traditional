"""
gvfa_traditional_combined.py
============================================================================
Combine GVFA hypervector encoding with traditional molecular descriptors
using a selectable COMBINATION/NORMALIZATION method and a selectable
REGRESSOR.

This single script consolidates every experiment that lived in
`GVFA_with_Traditional_feature.ipynb`. Instead of duplicating the long
GVFA pipeline in many cells, the shared parts are written once and the two
things that actually varied between experiments are exposed as command-line
inputs:

    1. --combine   how the traditional features are combined with the GVFA
                   embeddings, and how they are normalized.
    2. --regressor which model is trained on the combined features.

----------------------------------------------------------------------------
COMBINATION / NORMALIZATION METHODS  (--combine)
----------------------------------------------------------------------------
    concat              Raw concatenation. No scaling. (notebook cell 9)
    concat_scale_desc   StandardScaler on the 298 descriptors only, then
                        concatenate with raw GVFA. (cells 11, 13/14)
    concat_scale_both   StandardScaler the descriptors and the GVFA
                        embeddings separately, then concatenate. (cell 18)
    concat_scale_full   Concatenate raw, then StandardScaler the whole
                        feature space. Recommended for SVR/Ridge. (cell 17)
    concat_rp           Gaussian random-projection of the 298 descriptors up
                        to the GVFA dimension, then concatenate. (cell 21)
    concat_pca          PCA-reduce the GVFA embeddings to --pca-dim, then
                        concatenate with the raw descriptors. (cell 23)
    superposition       Project descriptors to the GVFA dimension through a
                        clean pipeline (impute -> robust scale -> random
                        projection), element-wise ADD to the GVFA embeddings,
                        then L2-normalize the rows (bundling). (cells 25-26)

----------------------------------------------------------------------------
REGRESSORS  (--regressor)
----------------------------------------------------------------------------
    xgb         XGBRegressor (the default used in most cells)
    svr_rbf     SVR with RBF kernel        (cell 17)
    svr_linear  SVR with linear kernel     (cell 17)
    ridge       Ridge regression           (cell 17)

----------------------------------------------------------------------------
EXAMPLES
----------------------------------------------------------------------------
    # Raw concatenation + XGBoost, sweep several HV dimensions
    python gvfa_traditional_combined.py --combine concat --regressor xgb \\
        --hv-dims 100 500 1000 2000 5000 10000

    # Scale both sides + XGBoost at HV=2000
    python gvfa_traditional_combined.py --combine concat_scale_both \\
        --regressor xgb --hv-dims 2000

    # Full-space scaling + Ridge
    python gvfa_traditional_combined.py --combine concat_scale_full \\
        --regressor ridge --hv-dims 2000

    # Superposition / bundling + XGBoost
    python gvfa_traditional_combined.py --combine superposition \\
        --regressor xgb --hv-dims 2000

The descriptor sets, the GVFA graph pipeline, and the metric reporting all
mirror the original notebook so results stay consistent.
============================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE MODE — must run BEFORE deepchem is imported anywhere.
#  DeepChem's CGCNNFeaturizer calls download_url() at IMPORT TIME (inside
#  deepchem/__init__.py → molnet → CGCNN). On NCI compute nodes there is no
#  outbound network, so the import crashes.
#
#  We patch the stdlib network primitives (urllib.request) that deepchem
#  ultimately calls. urllib.request has no import side effects, so patching it
#  first guarantees no network access when deepchem later imports via
#  `src.utilities`.
#
#  If a pre-staged copy of the requested file exists in offline_data/, it is
#  copied to the destination; otherwise the call becomes a harmless no-op
#  (an empty placeholder file is created and a warning is emitted).
# ══════════════════════════════════════════════════════════════════════════════
import os
import shutil
import urllib.request as _urlreq

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_DIR  = os.path.join(_PROJECT_ROOT, "offline_data")

os.environ["DEEPCHEM_DATA_DIR"] = _OFFLINE_DIR
os.makedirs(_OFFLINE_DIR, exist_ok=True)

_real_urlretrieve = _urlreq.urlretrieve

def _offline_urlretrieve(url, filename=None, *args, **kwargs):
    """Offline replacement for urllib.request.urlretrieve — never hits network."""
    name = url.split("/")[-1]

    if filename is None:
        filename = os.path.join(_OFFLINE_DIR, name)

    if os.path.exists(filename):
        return filename, None

    staged = os.path.join(_OFFLINE_DIR, name)
    if os.path.exists(staged) and os.path.abspath(staged) != os.path.abspath(filename):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        shutil.copy(staged, filename)
        return filename, None

    import warnings as _w
    _w.warn(f"[offline] network blocked; '{name}' not found in offline_data/. "
            f"Writing empty placeholder at {filename}.")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    open(filename, "a").close()
    return filename, None

_urlreq.urlretrieve = _offline_urlretrieve
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import copy
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

from rdkit import RDLogger

# Project-local imports (same as the notebook).
# NOTE: `src.utilities` does `import deepchem as dc` at module load, which
# triggers a network download unless the urlretrieve patch above is already
# installed. Do not move these imports above the OFFLINE MODE block.
from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from src import utilities
from models.graphcnnVSA_Binding_FULL import GraphCNN

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# 1. Traditional descriptor extraction
# ---------------------------------------------------------------------------
def build_traditional_features(train_set, test_set):
    """Build the 298-dimensional traditional descriptor set and alignment masks.

    298 = 123 RDKit descriptors + 128 Morgan fingerprint bits
          + 7 functional-group counts + 38 engineered structural features.

    Mirrors notebook cells 2-3 and the alignment logic in experiments/_shared.py.

    Every sub-frame is reset_index'd before concatenation so that pandas aligns
    rows by position rather than by whatever index a featurizer happens to
    produce.  The FULL (unfiltered) concat is used to build a boolean mask —
    True where a molecule has at least one valid (non-NaN) descriptor.  That
    mask is returned so the caller can apply it to the GVFA embeddings, which
    are built from the same CSV and therefore have one row per CSV molecule.

    Returns
    -------
    df298_train, df298_test : filtered DataFrames (invalid molecules removed)
    train_valid_mask, test_valid_mask : bool ndarray, shape [N_csv_train/test]
        Index these into the GVFA embedding arrays to align both sides.
    """
    print("Generating traditional descriptors ...")

    # .reset_index(drop=True) ensures position-based alignment in pd.concat
    # regardless of what index the individual featurizers produce.
    df123_train = utilities.generate123(train_set.smiles_canon).reset_index(drop=True)
    df123_test  = utilities.generate123(test_set.smiles_canon).reset_index(drop=True)

    df38_train  = utilities.generate_features38(train_set.smiles_canon).reset_index(drop=True)
    df38_test   = utilities.generate_features38(test_set.smiles_canon).reset_index(drop=True)

    df7_train   = utilities.get_functional_groups(train_set.smiles_canon).reset_index(drop=True)
    df7_test    = utilities.get_functional_groups(test_set.smiles_canon).reset_index(drop=True)

    df128_train = utilities.fingerprint(train_set.smiles_canon, 2, 128).reset_index(drop=True)
    df128_test  = utilities.fingerprint(test_set.smiles_canon, 2, 128).reset_index(drop=True)

    # Full (unfiltered) concat — one row per CSV molecule, NaN where failed.
    df298_full_train = pd.concat(
        [df123_train, df128_train, df7_train, df38_train], axis=1
    )
    df298_full_test = pd.concat(
        [df123_test, df128_test, df7_test, df38_test], axis=1
    )

    # Boolean mask: True = at least one descriptor is non-NaN (molecule is valid).
    # Length equals the FULL CSV row count so it can index into GVFA embeddings.
    train_valid = df298_full_train.notna().any(axis=1).values
    test_valid  = df298_full_test.notna().any(axis=1).values

    n_drop_tr = int((~train_valid).sum())
    n_drop_te = int((~test_valid).sum())
    if n_drop_tr:
        print(f"  [align] dropped {n_drop_tr} train molecules "
              f"(all-NaN descriptors — same rows will be dropped from GVFA)")
    if n_drop_te:
        print(f"  [align] dropped {n_drop_te} test molecules "
              f"(all-NaN descriptors — same rows will be dropped from GVFA)")

    df298_train = df298_full_train[train_valid].reset_index(drop=True)
    df298_test  = df298_full_test[test_valid].reset_index(drop=True)

    print(f"Traditional feature matrix: train {df298_train.shape}, "
          f"test {df298_test.shape}")
    return df298_train, df298_test, train_valid, test_valid


# ---------------------------------------------------------------------------
# 2. GVFA embedding extraction
# ---------------------------------------------------------------------------
def build_gvfa_embeddings(hv_dim,
                          train_csv,
                          test_csv,
                          train_valid_mask,
                          test_valid_mask,
                          num_layers=5,
                          delta=1,
                          equation=10,
                          graph_pooling_type="sum",
                          neighbor_pooling_type="sum",
                          device=None):
    """Run the GVFA pipeline and return aligned (train_emb, train_labels,
    test_emb, test_labels) as float32 numpy arrays.

    Graphs are rebuilt per HV dimension exactly like the notebook.

    The GVFA pipeline (ZINCLikeCSV) keeps every molecule that basic RDKit can
    parse, which may be more molecules than the descriptor pipeline accepts.
    We apply train_valid_mask / test_valid_mask — derived from the descriptor
    step — to select the same molecule subset in both feature sets.

    Parameters
    ----------
    train_csv, test_csv : str
        Paths to the same CSV files used for traditional features.
    train_valid_mask, test_valid_mask : bool ndarray
        Returned by build_traditional_features.  Length must equal the number
        of rows in the respective CSV (i.e. the GVFA embedding count).
    """
    if device is None:
        device = torch.device("cpu")

    # Load from the same CSVs as the descriptor pipeline so molecule ordering
    # is identical and the boolean masks index the correct rows.
    train_data, test_data = load_data(
        dataset="new",
        train_path=train_csv,
        test_path=test_csv,
    )

    train_graphs = create_graph_list(train_data)
    test_graphs  = create_graph_list(test_data)

    # Use copies for the projection step to avoid mutating originals.
    test_HVs  = VSA_conversion(copy.deepcopy(test_graphs),  hv_dim)
    train_HVs = VSA_conversion(copy.deepcopy(train_graphs), hv_dim)

    in_dim = test_HVs[0].node_features.shape[1]
    model = GraphCNN(
        in_dim,
        num_layers,
        delta,
        graph_pooling_type,
        neighbor_pooling_type,
        device,
        equation,
    )

    train_emb, train_labels = getEmbedding(model, device, train_HVs)
    test_emb,  test_labels  = getEmbedding(model, device, test_HVs)

    # Convert to float32 numpy for masking and downstream use.
    gvfa_tr = _to_numpy_f32(train_emb.squeeze(0))   # [N_full_train, D]
    gvfa_te = _to_numpy_f32(test_emb.squeeze(0))    # [N_full_test,  D]
    y_tr    = np.asarray(train_labels.cpu()).ravel().astype(np.float32)
    y_te    = np.asarray(test_labels.cpu()).ravel().astype(np.float32)

    n_full_tr, n_full_te = gvfa_tr.shape[0], gvfa_te.shape[0]
    n_keep_tr = int(train_valid_mask.sum())
    n_keep_te = int(test_valid_mask.sum())

    # Apply descriptor-derived masks to align GVFA rows with descriptor rows.
    # Fallback (mask length mismatch): keep the first n_keep rows by position —
    # safe because both pipelines read the same CSV in the same order.
    if len(train_valid_mask) == n_full_tr:
        gvfa_tr = gvfa_tr[train_valid_mask]
        y_tr    = y_tr[train_valid_mask]
    else:
        print(f"  [align] train mask length {len(train_valid_mask)} != "
              f"GVFA rows {n_full_tr}; keeping first {n_keep_tr} rows")
        gvfa_tr = gvfa_tr[:n_keep_tr]
        y_tr    = y_tr[:n_keep_tr]

    if len(test_valid_mask) == n_full_te:
        gvfa_te = gvfa_te[test_valid_mask]
        y_te    = y_te[test_valid_mask]
    else:
        print(f"  [align] test mask length {len(test_valid_mask)} != "
              f"GVFA rows {n_full_te}; keeping first {n_keep_te} rows")
        gvfa_te = gvfa_te[:n_keep_te]
        y_te    = y_te[:n_keep_te]

    print(f"  GVFA aligned: train {gvfa_tr.shape}  test {gvfa_te.shape}")
    return gvfa_tr, y_tr, gvfa_te, y_te


# ---------------------------------------------------------------------------
# 3. Combination / normalization strategies
# ---------------------------------------------------------------------------
def _to_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _make_clean_pipeline(rp_dim, use_knn=False):
    """Impute -> robust scale -> random projection pipeline for the
    superposition path (notebook cell 25)."""
    imputer = (KNNImputer(n_neighbors=5, weights="distance")
               if use_knn else SimpleImputer(strategy="median"))
    pipe = Pipeline([
        ("impute", imputer),
        ("clipper", RobustScaler(with_centering=True, with_scaling=True)),
        ("rp", GaussianRandomProjection(n_components=rp_dim, random_state=42)),
    ])

    def _prep(df):
        return df.replace([np.inf, -np.inf], np.nan)

    pipe._prep = _prep
    return pipe


def combine_features(method,
                     df298_train, df298_test,
                     train_emb, test_emb,
                     hv_dim,
                     pca_dim=100,
                     rp_seed=42):
    """Return (X_train, X_test) as numpy float32 arrays according to the
    chosen combination/normalization method."""

    gvfa_train = _to_numpy_f32(train_emb)
    gvfa_test = _to_numpy_f32(test_emb)
    desc_train = df298_train.to_numpy(dtype=np.float32)
    desc_test = df298_test.to_numpy(dtype=np.float32)

    if method == "concat":
        # cell 9 : raw concatenation
        X_train = np.concatenate([desc_train, gvfa_train], axis=1)
        X_test = np.concatenate([desc_test, gvfa_test], axis=1)

    elif method == "concat_scale_desc":
        # cells 11 / 13 / 14 : scale descriptors only
        scaler = StandardScaler().fit(desc_train)
        X_train = np.concatenate([scaler.transform(desc_train), gvfa_train], axis=1)
        X_test = np.concatenate([scaler.transform(desc_test), gvfa_test], axis=1)

    elif method == "concat_scale_both":
        # cell 18 : scale descriptors and GVFA separately
        sc_df = StandardScaler(with_mean=True, with_std=True)
        sc_hv = StandardScaler(with_mean=True, with_std=True)
        d_tr = sc_df.fit_transform(desc_train)
        d_te = sc_df.transform(desc_test)
        h_tr = sc_hv.fit_transform(gvfa_train)
        h_te = sc_hv.transform(gvfa_test)
        X_train = np.concatenate([d_tr, h_tr], axis=1)
        X_test = np.concatenate([d_te, h_te], axis=1)

    elif method == "concat_scale_full":
        # cell 17 : raw concat, then scale the whole feature space
        X_train = np.concatenate([desc_train, gvfa_train], axis=1)
        X_test = np.concatenate([desc_test, gvfa_test], axis=1)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        scaler_full = StandardScaler()
        X_train = scaler_full.fit_transform(X_train)
        X_test = scaler_full.transform(X_test)

    elif method == "concat_rp":
        # cell 21 : random-project descriptors to HV dim, then concat
        proj = GaussianRandomProjection(n_components=hv_dim, random_state=rp_seed)
        d_tr = np.nan_to_num(desc_train, nan=0.0, posinf=0.0, neginf=0.0)
        d_te = np.nan_to_num(desc_test, nan=0.0, posinf=0.0, neginf=0.0)
        d_tr = proj.fit_transform(d_tr).astype(np.float32)
        d_te = proj.transform(d_te).astype(np.float32)
        X_train = np.concatenate([d_tr, gvfa_train], axis=1)
        X_test = np.concatenate([d_te, gvfa_test], axis=1)

    elif method == "concat_pca":
        # cell 23 : PCA-reduce GVFA embeddings, then concat with descriptors
        pca = PCA(n_components=pca_dim, random_state=42)
        h_tr = pca.fit_transform(gvfa_train).astype(np.float32)
        h_te = pca.transform(gvfa_test).astype(np.float32)
        X_train = np.concatenate([desc_train, h_tr], axis=1)
        X_test = np.concatenate([desc_test, h_te], axis=1)

    elif method == "superposition":
        # cells 25-26 : impute->robust->random-project descriptors to HV dim,
        # element-wise add to GVFA, then L2-normalize rows
        pipe = _make_clean_pipeline(rp_dim=hv_dim, use_knn=False)
        Xtr_df = pipe._prep(df298_train).astype(np.float32)
        Xte_df = pipe._prep(df298_test).astype(np.float32)
        d_tr = pipe.fit_transform(Xtr_df).astype(np.float32)
        d_te = pipe.transform(Xte_df).astype(np.float32)

        S_train = d_tr + gvfa_train
        S_test = d_te + gvfa_test
        S_train /= (np.linalg.norm(S_train, axis=1, keepdims=True) + 1e-8)
        S_test /= (np.linalg.norm(S_test, axis=1, keepdims=True) + 1e-8)
        X_train, X_test = S_train, S_test

    else:
        raise ValueError(f"Unknown combine method: {method}")

    return X_train.astype(np.float32), X_test.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Regressors
# ---------------------------------------------------------------------------
def build_regressor(name):
    """Return an unfitted regressor by name."""
    if name == "xgb":
        return XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist",   # use "gpu_hist" if a GPU is available
        )
    if name == "svr_rbf":
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    if name == "svr_linear":
        return SVR(kernel="linear", C=1.0, epsilon=0.1)
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=42)
    raise ValueError(f"Unknown regressor: {name}")


def fit_and_predict(regressor_name, X_train, y_train, X_test, y_test):
    """Fit the chosen regressor and return predictions on the test set."""
    model = build_regressor(regressor_name)

    # Linear / SVR models are sensitive to NaN/inf and need 1-D labels.
    y_train = np.nan_to_num(np.asarray(y_train).ravel(),
                            nan=0.0, posinf=1e6, neginf=-1e6)
    y_test = np.nan_to_num(np.asarray(y_test).ravel(),
                           nan=0.0, posinf=1e6, neginf=-1e6)

    if regressor_name == "xgb":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model.predict(X_test), y_test


# ---------------------------------------------------------------------------
# 5. Main driver
# ---------------------------------------------------------------------------
def run_experiment(args):
    device = torch.device("cpu")

    # ── Step 1: traditional descriptors (done once, before the HV loop) ──────
    # build_traditional_features also returns boolean masks that identify which
    # CSV rows have at least one valid descriptor.  These masks are used in
    # Step 2 to drop the same molecules from the GVFA embeddings so both
    # feature matrices have identical row counts.
    train_set = pd.read_csv(args.train_csv)
    test_set  = pd.read_csv(args.test_csv)
    df298_train, df298_test, train_valid_mask, test_valid_mask = \
        build_traditional_features(train_set, test_set)

    all_results = []

    for hv_dim in args.hv_dims:
        print("\n" + "=" * 70)
        print(f"HV dimension = {hv_dim} | combine = {args.combine} | "
              f"regressor = {args.regressor}")
        print("=" * 70)

        # ── Step 2: GVFA embeddings aligned to the descriptor molecule set ───
        train_emb, train_labels, test_emb, test_labels = build_gvfa_embeddings(
            hv_dim,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            train_valid_mask=train_valid_mask,
            test_valid_mask=test_valid_mask,
            num_layers=args.num_layers,
            delta=args.delta,
            equation=args.equation,
            graph_pooling_type=args.graph_pooling,
            neighbor_pooling_type=args.neighbor_pooling,
            device=device,
        )

        # ── Step 3: combine + normalize ──────────────────────────────────────
        X_train, X_test = combine_features(
            args.combine,
            df298_train, df298_test,
            train_emb, test_emb,
            hv_dim,
            pca_dim=args.pca_dim,
            rp_seed=args.rp_seed,
        )
        print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

        # ── Step 4: train + predict ───────────────────────────────────────────
        preds, y_test = fit_and_predict(
            args.regressor, X_train, train_labels, X_test, test_labels
        )

        # Report metrics using the project's helper (same as notebook).
        tag = f"{args.regressor}_298 {args.combine} GVFA({hv_dim})"
        metrics = utilities.get_errors1(y_test, preds, tag)
        metrics["Descriptors_Detail"] = (
            "123 features + 128 fingerprint + 7 f_group + 38 fe features"
        )
        print(metrics)
        all_results.append(metrics)

    # Aggregate and optionally save.
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        print("\n" + "=" * 70)
        print("ALL RESULTS")
        print("=" * 70)
        print(results_df)
        if args.output_csv:
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nSaved results to {args.output_csv}")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine GVFA encoding with traditional features. "
                    "Select the combination/normalization method and the "
                    "regressor from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # The two selectable axes requested.
    p.add_argument(
        "--combine", default="concat",
        choices=[
            "concat",
            "concat_scale_desc",
            "concat_scale_both",
            "concat_scale_full",
            "concat_rp",
            "concat_pca",
            "superposition",
        ],
        help="How to combine/normalize traditional features with GVFA.",
    )
    p.add_argument(
        "--regressor", default="xgb",
        choices=["xgb", "svr_rbf", "svr_linear", "ridge"],
        help="Regressor to train on the combined features.",
    )

    # GVFA dimensions to sweep.
    p.add_argument(
        "--hv-dims", type=int, nargs="+",
        default=[100, 500, 1000, 2000, 5000, 10000],
        help="One or more GVFA hypervector dimensions to evaluate.",
    )

    # Method-specific knobs.
    p.add_argument("--pca-dim", type=int, default=100,
                   help="PCA components for the concat_pca method.")
    p.add_argument("--rp-seed", type=int, default=42,
                   help="Random seed for random-projection methods.")

    # Data paths.
    p.add_argument("--train-csv",
                   default="final_data/final_unique_train_fixed.csv",
                   help="Training CSV with a smiles_canon column.")
    p.add_argument("--test-csv",
                   default="final_data/final_unique_test.csv",
                   help="Test CSV with a smiles_canon column.")
    p.add_argument("--output-csv", default=None,
                   help="Optional path to save the results table.")

    # GVFA / GraphCNN hyperparameters (defaults match the notebook).
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--delta", type=int, default=1)
    p.add_argument("--equation", type=int, default=10)
    p.add_argument("--graph-pooling", default="sum",
                   choices=["sum", "average"])
    p.add_argument("--neighbor-pooling", default="sum",
                   choices=["sum", "average", "max"])

    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())