"""
GVFA sweep — GVFA-only or GVFA + 298 traditional descriptors.

Usage
-----
    # Combined (default): GVFA embeddings + scaled df298 features
    python GVFA_STD_298.py --mode combined --seeds 0,1,2,3,4

    # GVFA-only: embeddings without traditional descriptors
    python GVFA_STD_298.py --mode gvfa --seeds 0,1,2,3,4 --dims 100,500,1000

Results are printed and saved to results_gvfa_only.csv or results_gvfa_298.csv.
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from models.graphcnnVSA_Binding_FULL_edge import GraphCNN
from src.create_graphs import create_graph_list
from src.embeddings import getEmbedding
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion, project_with_vsa

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GVFA sweep (GVFA-only or combined)")
parser.add_argument(
    "--mode", type=str, default="combined", choices=["gvfa", "combined"],
    help="'gvfa' = GVFA embeddings only; 'combined' = GVFA + scaled df298 (default)",
)
parser.add_argument(
    "--seeds", type=str, default="0",
    help="Comma-separated seed values (default: 0), e.g. --seeds 0,1,2,3,4",
)
parser.add_argument(
    "--dims", type=str, default="100,500,1000,2000,5000,10000",
    help="Comma-separated HV dimensions (default: 100,500,1000,2000,5000,10000)",
)
args = parser.parse_args()

MODE          = args.mode
SEEDS         = [int(s.strip()) for s in args.seeds.split(",")]
HV_Dimentions = [int(d.strip()) for d in args.dims.split(",")]
RESULTS_CSV   = "results_gvfa_only.csv" if MODE == "gvfa" else "results_gvfa_298.csv"

print(f"Mode        : {MODE}")
print(f"Seeds       : {SEEDS}")
print(f"HV dims     : {HV_Dimentions}")

# ── Traditional features (combined mode only) ─────────────────────────────────
df_torch_train = None
df_torch_test  = None

if MODE == "combined":
    if os.path.exists("offline_data/df298_train.parquet"):
        print("Loading cached df298 ...")
        df298_train = pd.read_parquet("offline_data/df298_train.parquet")
        df298_test  = pd.read_parquet("offline_data/df298_test.parquet")
    else:
        raise FileNotFoundError(
            "offline_data/df298_train.parquet not found. "
            "Run save_298_features.py first."
        )

    scaler_298 = StandardScaler()
    scaler_298.fit(df298_train.values)

    df_torch_train = torch.from_numpy(
        scaler_298.transform(df298_train.values).astype(np.float32)
    )
    df_torch_test = torch.from_numpy(
        scaler_298.transform(df298_test.values).astype(np.float32)
    )


def get_errors1(y_true, y_pred, model_name="Model"):
    err_mae  = round(mae(y_true, y_pred), 4)
    err_rmse = round(np.sqrt(mse(y_true, y_pred)), 4)
    err_r2   = round(r2(y_true, y_pred), 4)
    err_mse  = round(mse(y_true, y_pred), 4)
    results = np.column_stack([model_name, err_mae, err_mse, err_rmse, err_r2])
    return pd.DataFrame(results, columns=["Model_Name", "MAE", "MSE", "RMSE", "R2"])


# ── Graph topology (built once, shared across all seeds/dims) ─────────────────
num_layers            = 5
delta_eq1             = 1
equation_eq1          = 10
graph_pooling_type    = "sum"
neighbor_pooling_type = "sum"
device                = torch.device("cpu")

train_data, test_data = load_data(dataset="new")
print(f"Train graphs: {len(train_data)}  Test graphs: {len(test_data)}")

train_graphs = create_graph_list(train_data)
test_graphs  = create_graph_list(test_data)

# Build neighbors / edge_mat ONCE — topology is the same for every (dim, seed).
# Passing new_dim=None skips projection and only builds the graph structure.
VSA_conversion(train_graphs, new_dim=None)
VSA_conversion(test_graphs,  new_dim=None)

# Save original node features as float32 clones so we can restore them
# before each projection without rebuilding graphs from scratch.
# This replaces deepcopy: only tensors are cloned, not the full graph objects.
train_orig_features = [g.node_features.clone().to(torch.float32) for g in train_graphs]
test_orig_features  = [g.node_features.clone().to(torch.float32) for g in test_graphs]

# ── Sweep over (dim, seed) ────────────────────────────────────────────────────
all_results = []

for HV_Dimention in HV_Dimentions:
    for seed in SEEDS:
        print(f"\n=== dim={HV_Dimention}  seed={seed} ===")

        # Restore original atom features before each projection.
        # project_with_vsa modifies node_features in-place, so this is required.
        for g, orig in zip(train_graphs, train_orig_features):
            g.node_features = orig.clone()
        for g, orig in zip(test_graphs, test_orig_features):
            g.node_features = orig.clone()

        # Project atom feature vectors → hypervectors of size HV_Dimention.
        # Topology (neighbors / edge_mat) was already built above; skip rebuild.
        train_HVs = project_with_vsa(train_graphs, HV_Dimention, seed=seed)
        test_HVs  = project_with_vsa(test_graphs,  HV_Dimention, seed=seed)

        # Edge-conditioned GVFA: concat(neighbor HV, edge HV) then reduce to D.
        # edge_attr is [E, 5] from create_graphs (bond type, conjugated, ring, length, stereo).
        edge_feat_dim = getattr(train_HVs[0], "edge_attr", None)
        edge_feat_dim = edge_feat_dim.shape[1] if edge_feat_dim is not None else 5

        model_eq1 = GraphCNN(
            train_HVs[0].node_features.shape[1],
            num_layers, delta_eq1,
            graph_pooling_type, neighbor_pooling_type,
            device, equation_eq1,
            edge_feat_dim=edge_feat_dim,
            rng_seed=seed,
        )

        train_embeddings, train_labels = getEmbedding(model_eq1, device, train_HVs)
        test_embeddings,  test_labels  = getEmbedding(model_eq1, device, test_HVs)

        train_embeddings = train_embeddings.squeeze(0)
        test_embeddings  = test_embeddings.squeeze(0)

        if MODE == "gvfa":
            X_train = train_embeddings
            X_test  = test_embeddings
        else:
            X_train = torch.cat([df_torch_train, train_embeddings], dim=1)
            X_test  = torch.cat([df_torch_test,  test_embeddings],  dim=1)
        print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

        xgb = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
        )
        xgb.fit(X_train, train_labels, eval_set=[(X_test, test_labels)], verbose=False)

        pred_xgb = xgb.predict(X_test)

        if MODE == "gvfa":
            model_label = f"XGB GVFA-only(dim={HV_Dimention} seed={seed})"
            desc_detail = "GVFA embeddings only"
        else:
            model_label = f"XGB_298 GVFA(dim={HV_Dimention} seed={seed})"
            desc_detail = "125 features + 128 fingerprint 7 f_group+38 fe features"

        row = get_errors1(test_labels, pred_xgb, model_label)
        row["seed"] = seed
        row["dim"]  = HV_Dimention
        row["mode"] = MODE
        row["Descriptors_Detail"] = desc_detail
        all_results.append(row)
        print(row)

        del model_eq1, train_embeddings, train_labels, test_embeddings, test_labels
        del X_train, X_test, xgb, pred_xgb
        gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
summary = pd.concat(all_results, ignore_index=True)
print("\n=== Full Results ===")
print(summary.to_string(index=False))
summary.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved to {RESULTS_CSV}")
