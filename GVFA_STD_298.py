from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import pandas as pd
from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
from src import utilities

if os.path.exists("offline_data/df298_train.parquet"):
    print("Loading cached df298 ...")
    df298_train = pd.read_parquet("offline_data/df298_train.parquet") # 17929
    df298_test  = pd.read_parquet("offline_data/df298_test.parquet")
    # train_valid = np.load("offline_data/train_valid_mask.npy")
    # test_valid  = np.load("offline_data/test_valid_mask.npy")
    

HV_Dimentions = [200]

scaler_298 = StandardScaler()
scaler_298.fit(df298_train.values)   # each column: its own mean/std


for HV_Dimention in HV_Dimentions:

    train_data, test_data = load_data(dataset="new")
    # print(train_data[0].edge_attr)
    print(len(train_data))
    print(len(test_data))

    # train_graphs = create_graph_list(train_data)
    # test_graphs = create_graph_list(test_data)


    num_layers = 5
    delta_eq1 = 1
    equation_eq1 = 10
    graph_pooling_type = 'sum'  # sum, average
    neighbor_pooling_type = 'sum' # sum, average, max
    device = 1  # help='if delta is 1 will be the model with binding, if 0 model will have be without binding (default: 1)'
    device = torch.device('cpu')
    
    train_graphs = create_graph_list(train_data)
    test_graphs = create_graph_list(test_data)
    ts_graph = test_graphs.copy()
    tr_graph = train_graphs.copy()

    test_HVs = VSA_conversion(ts_graph, HV_Dimention)
    train_HVs = VSA_conversion(tr_graph, HV_Dimention)

    model_eq1 = GraphCNN(test_HVs[0].node_features.shape[1], num_layers, delta_eq1, graph_pooling_type, neighbor_pooling_type, device, equation_eq1) #.to(device)
    train_embeddings_eq1, train_labels_eq1 = getEmbedding(model_eq1, device, train_HVs)
    test_embeddings_eq1, test_labels_eq1 = getEmbedding(model_eq1, device, test_HVs)

    train_embeddings_eq1 = train_embeddings_eq1.squeeze(0)
    test_embeddings_eq1 = test_embeddings_eq1.squeeze(0)

    df298_train_scaled = scaler_298.transform(df298_train.values)
    df298_test_scaled  = scaler_298.transform(df298_test.values) 


    df_torch_train = torch.from_numpy(df298_train_scaled.astype(np.float32))
    df_torch_test  = torch.from_numpy(df298_test_scaled.astype(np.float32))


    X_train = torch.cat([df_torch_train, train_embeddings_eq1], axis=1)

    # X_train = pd.concat([df_t, train_embeddings_eq1], axis=1)
    X_test = torch.cat([df_torch_test, test_embeddings_eq1], axis=1)

    print(X_train.shape)
    print(X_test.shape)

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
        tree_method="hist"   # fast on CPU; use "gpu_hist" if you have GPU
    )

    xgb.fit(
        X_train, train_labels_eq1,
        eval_set=[(X_test, test_labels_eq1)],
        # early_stopping_rounds=100,
        verbose=False
    )

    pred_xgb = xgb.predict(X_test)
    
    xgb_298=utilities.get_errors1(test_labels_eq1,pred_xgb,f"XGB_298 concatinate GVFA({HV_Dimention})")
    xgb_298['Descriptors_Detail']='125 features + 128 fingerprint 7 f_group+38 fe features'
    print(xgb_298)