"""
exp9_featurewise_norm.py
========================
Type-aware, column-wise normalization of the 298-feature block before
concatenation with GVFA embeddings.

Strategy
--------
The 298-dim descriptor block is assembled as:
  cols   0–122  : 123 RDKit physicochemical descriptors  (mixed: continuous + a few binary)
  cols 123–250  : 128-bit Morgan fingerprint             (all binary  0/1)
  cols 251–257  : 7 functional-group indicators          (all binary  0/1)
  cols 258–295  : 38 engineered features                 (mixed: counts + binary flags)

Normalization rules applied per column:
  BINARY    (only values in {0, 1, NaN})  → remap to {-1, +1}
  NEAR-CONST (std < 1e-3 after imputing)  → drop column entirely
  CONTINUOUS (everything else)            → tanh( (x - median) / (IQR + ε) )

The GVFA block is NOT scaled here — it is already in a controlled range
thanks to the clip(κ=1) normalization inside GVFA.

Usage
-----
    from experiments.exp9_featurewise_norm import run
    run()
or via the main menu:  python run_experiments.py  → [9]
"""

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

from experiments._shared import (
    build_gvfa_embeddings,
    fit_xgb_and_report,
    to_numpy,
    clean_nan,
    DESC_DETAIL,
)
from src import utilities

# ── tuneable ──────────────────────────────────────────────────────────────────
HV_DIM          = 2000          # sweet-spot from prior ablation
NEAR_CONST_THR  = 1e-3          # columns with std < this are dropped
IQR_EPS         = 1e-6          # avoid divide-by-zero in tanh scaling
TANH_SCALE      = 3.0           # controls how aggressively tanh compresses outliers
                                 # tanh(3) ≈ 0.995, so ±3 IQR → ±1


# ── column-type detection ─────────────────────────────────────────────────────

def _detect_column_types(df: pd.DataFrame):
    """
    Return three index arrays:
      binary_cols      – columns whose unique non-NaN values ⊆ {0, 1}
      near_const_cols  – columns with std < NEAR_CONST_THR (after imputing)
      continuous_cols  – everything else

    Detection is done on the TRAINING dataframe only.
    """
    arr = df.values.astype(np.float32)

    binary_cols    = []
    near_const_cols = []
    continuous_cols = []

    for j in range(arr.shape[1]):
        col = arr[:, j]
        col_valid = col[~np.isnan(col)]

        if len(col_valid) == 0:
            near_const_cols.append(j)
            continue

        unique_vals = set(np.unique(col_valid))
        is_binary   = unique_vals.issubset({0.0, 1.0})

        if is_binary:
            binary_cols.append(j)
        elif col_valid.std() < NEAR_CONST_THR:
            near_const_cols.append(j)
        else:
            continuous_cols.append(j)

    return (
        np.array(binary_cols,     dtype=int),
        np.array(near_const_cols, dtype=int),
        np.array(continuous_cols, dtype=int),
    )


# ── the normalizer ────────────────────────────────────────────────────────────

class TypeAwareNormalizer:
    """
    Fit on training data, transform train and test.

    After fit():
        self.binary_cols      – indices remapped to {-1, +1}
        self.dropped_cols     – indices removed (near-constant)
        self.continuous_cols  – indices tanh-scaled
        self.keep_cols        – ordered list of columns surviving the drop step
    """

    def __init__(self):
        self._fitted = False

    def fit(self, df_train: pd.DataFrame):
        arr = df_train.values.astype(np.float32)

        # 1. NaN-impute with median (needed for std calculation)
        self._imputer = SimpleImputer(strategy="median")
        arr_imp = self._imputer.fit_transform(arr)

        # 2. Detect column types
        binary_idx, drop_idx, cont_idx = _detect_column_types(
            pd.DataFrame(arr_imp)       # pass imputed values for type detection
        )

        self._binary_idx = binary_idx
        self._drop_idx   = drop_idx
        self._cont_idx   = cont_idx

        # 3. Compute median + IQR for continuous columns (on imputed training data)
        cont_data          = arr_imp[:, cont_idx]
        self._cont_median  = np.median(cont_data, axis=0)
        q75, q25           = np.percentile(cont_data, [75, 25], axis=0)
        self._cont_iqr     = q75 - q25 + IQR_EPS

        # 4. Build keep_cols = binary + continuous (no near-const), in original order
        keep = sorted(set(binary_idx.tolist()) | set(cont_idx.tolist()))
        self._keep_cols = np.array(keep, dtype=int)

        self._fitted = True

        n_total  = arr.shape[1]
        n_bin    = len(binary_idx)
        n_drop   = len(drop_idx)
        n_cont   = len(cont_idx)
        print(f"  [TypeAwareNorm] total={n_total}  binary={n_bin}  "
              f"dropped(near-const)={n_drop}  continuous={n_cont}  "
              f"→ kept={len(keep)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "call fit() first"
        arr     = df.values.astype(np.float32)
        arr_imp = self._imputer.transform(arr)

        out = np.zeros_like(arr_imp)

        # binary: {0,1} → {-1,+1}
        if len(self._binary_idx):
            b = arr_imp[:, self._binary_idx]
            b = np.where(np.isnan(b), 0.0, b)          # NaN → neutral 0 → -1 after remap
            out[:, self._binary_idx] = 2.0 * b - 1.0   # 0→-1, 1→+1

        # continuous: tanh( (x - median) / (IQR * scale) )
        if len(self._cont_idx):
            c = arr_imp[:, self._cont_idx]
            c = (c - self._cont_median) / (self._cont_iqr * TANH_SCALE)
            out[:, self._cont_idx] = np.tanh(c)

        # keep only non-dropped columns
        return out[:, self._keep_cols].astype(np.float32)


# ── main experiment ───────────────────────────────────────────────────────────

def run():
    print(f"\n[Exp 9] Type-aware feature-wise normalization | HV dim = {HV_DIM}")
    print(f"        binary→{{-1,+1}}  near-const→drop  continuous→tanh(IQR)\n")

    from experiments._shared import (
        load_traditional_features_aligned,
        build_gvfa_embeddings_aligned,
    )

    # ── traditional features + valid masks ───────────────────────────────────
    df298_train, df298_test, \
        y_train, y_test, \
        train_mask, test_mask = load_traditional_features_aligned()

    norm = TypeAwareNormalizer()
    norm.fit(df298_train)

    desc_tr = norm.transform(df298_train)
    desc_te = norm.transform(df298_test)

    print(f"  Descriptor block shape after norm: train={desc_tr.shape}  test={desc_te.shape}")

    # ── GVFA embeddings aligned via same mask ─────────────────────────────────
    gvfa_tr, gvfa_te, y_train, y_test = build_gvfa_embeddings_aligned(
        hv_dim          = HV_DIM,
        n_train_keep    = desc_tr.shape[0],
        n_test_keep     = desc_te.shape[0],
        train_valid_mask= train_mask,
        test_valid_mask = test_mask,
    )

    # ── concatenate ──────────────────────────────────────────────────────────
    X_train = np.concatenate([desc_tr, gvfa_tr], axis=1)
    X_test  = np.concatenate([desc_te, gvfa_te], axis=1)

    X_train = clean_nan(X_train)
    X_test  = clean_nan(X_test)

    print(f"  Final X_train shape: {X_train.shape}")
    print(f"  Final X_test  shape: {X_test.shape}\n")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    result = fit_xgb_and_report(
        X_train, y_train.ravel(),
        X_test,  y_test.ravel(),
        label=f"XGB_TypeAwareNorm+GVFA(D={HV_DIM})",
    )

    # ── summary of what was done ─────────────────────────────────────────────
    print("\n  Column breakdown:")
    print(f"    Binary (→{{-1,+1}})     : {len(norm._binary_idx):>4d} cols")
    print(f"    Near-constant (dropped): {len(norm._drop_idx):>4d} cols")
    print(f"    Continuous (→tanh/IQR) : {len(norm._cont_idx):>4d} cols")
    print(f"    Kept total             : {len(norm._keep_cols):>4d} cols")

    return result


# ── allow direct execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    run()