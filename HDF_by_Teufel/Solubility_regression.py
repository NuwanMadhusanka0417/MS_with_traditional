"""
Encode + regress across (dimension, seed) combinations, with memory
instrumentation and explicit cleanup to prevent cross-iteration
accumulation.

Molecules are encoded in small batches so peak RAM stays bounded at high
dimensions (e.g. 10 000-D).  Each molecule's fingerprint depends only on
its own graph, so chunked encoding is numerically identical to encoding
the full list in one call.

Usage:
    python Solubility_regression.py
"""

import gc
import os

import numpy as np
import pandas as pd
import psutil

from hyper_fingerprints import Encoder
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


TRAIN_CSV = "../final_data/final_unique_train_fixed.csv"
TEST_CSV = "../final_data/final_unique_test.csv"
TARGET_COL = "logS"
SMILES_COL = "SMILES"
SEEDS = [0, 1, 2, 3, 4]
DIMENSIONS = [1000, 2000, 5000, 10000]

# Reuse saved encodings across runs (same dim / seed / split -> same file).
CACHE_DIR = "cached_hdf_features"

_process = psutil.Process()


def peak_mem_gb():
    """Current RSS memory usage of this process, in GB."""
    return _process.memory_info().rss / (1024 ** 3)


def encode_batch_size(dimension: int) -> int:
    """Smaller batches at higher dimension to cap encoder working memory."""
    if dimension <= 2000:
        return 512
    if dimension <= 5000:
        return 256
    return 128


def encode_in_chunks(encoder, smiles_list, batch_size: int) -> np.ndarray:
    """Encode a SMILES list in chunks; results match a single batch encode."""
    parts = []
    n = len(smiles_list)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        parts.append(encoder.encode(smiles_list[start:end]))
        gc.collect()
    return np.vstack(parts)


def cache_path(dimension: int, seed: int, split: str) -> str:
    return os.path.join(CACHE_DIR, f"hdf_dim{dimension}_seed{seed}_{split}.npy")


def load_or_encode(encoder, smiles_list, dimension: int, seed: int, split: str) -> np.ndarray:
    """Load cached encodings or compute and save them."""
    path = cache_path(dimension, seed, split)
    if os.path.exists(path):
        print(f"  Loading cached features: {path}")
        return np.load(path)

    batch_size = encode_batch_size(dimension)
    print(f"  Encoding {split} in chunks of {batch_size} ...")
    X = encode_in_chunks(encoder, smiles_list, batch_size)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, X)
    return X


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
    smiles = df[SMILES_COL].tolist()
    y = df[TARGET_COL].to_numpy(dtype=float)
    print(f"{csv_path}: loaded {len(smiles)} molecules.")
    return smiles, y


def detect_atom_vocabulary(smiles_list):
    from rdkit import Chem

    vocab = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            vocab.add(atom.GetSymbol())
    return sorted(vocab)


train_smiles, y_train = load_dataset(TRAIN_CSV)
test_smiles, y_test = load_dataset(TEST_CSV)
atom_vocab = detect_atom_vocabulary(train_smiles + test_smiles)
print(f"[mem] after data load: {peak_mem_gb():.2f} GB")


def build_models():
    """Fresh model instances each call -- never reuse a fitted model
    across iterations, since fitted attributes can retain references
    to training data."""
    return {
        "Ridge": RidgeCV(
            alphas=np.logspace(-6, 4, 80),
            cv=5,
            scoring="neg_mean_squared_error",
            fit_intercept=True,
        ),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


all_results = []
peak_mem_seen = peak_mem_gb()

for dim in DIMENSIONS:
    for seed in SEEDS:
        print(f"\n=== dim={dim} seed={seed} ===")
        print(f"[mem] loop start: {peak_mem_gb():.2f} GB")

        encoder = Encoder(
            dimension=dim,
            depth=3,
            atom_types=atom_vocab,
            seed=seed,
            normalize=False,
            backend="auto",
        )

        X_train = load_or_encode(encoder, train_smiles, dim, seed, "train")
        X_test = load_or_encode(encoder, test_smiles, dim, seed, "test")
        peak_mem_seen = max(peak_mem_seen, peak_mem_gb())
        print(
            f"[mem] after encode: {peak_mem_gb():.2f} GB  "
            f"(X_train {X_train.nbytes / 1e9:.3f} GB, "
            f"X_test {X_test.nbytes / 1e9:.3f} GB)"
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        peak_mem_seen = max(peak_mem_seen, peak_mem_gb())
        print(f"[mem] after scaling: {peak_mem_gb():.2f} GB")

        del X_train, X_test

        models = build_models()
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            all_results.append((dim, seed, name, rmse, r2))
            peak_mem_seen = max(peak_mem_seen, peak_mem_gb())
            print(
                f"{name:10s} dim={dim} seed={seed}  "
                f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  "
                f"[mem] {peak_mem_gb():.2f} GB"
            )

            del model

        del X_train_s, X_test_s, scaler, encoder, models
        gc.collect()
        print(f"[mem] after cleanup: {peak_mem_gb():.2f} GB")

results_df = pd.DataFrame(
    all_results, columns=["dimension", "seed", "model", "RMSE", "R2"]
)
results_df.to_csv("sweep_results.csv", index=False)
print(f"\nPeak memory observed during run: {peak_mem_seen:.2f} GB")
print(results_df)
