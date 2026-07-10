from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion
from src.embeddings import getEmbedding
from models.graphcnnVSA_Binding_FULL import GraphCNN
import torch
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors,rdMolDescriptors

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen

### Importing the required library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MolStandardize
import joblib
from src import utilities

train_set=pd.read_csv('final_data/final_unique_train_fixed.csv')
test_set=pd.read_csv('final_data/final_unique_test.csv')
print(len(train_set))
print(len(test_set))
train_smiles_list=train_set[['smiles_canon']]
test_smiles_list=test_set[['smiles_canon']]


### Generate 4 descriptors ....
df4_train=utilities.generate4(train_set.smiles_canon)
df4_test=utilities.generate4(test_set.smiles_canon)
print(len(df4_train))
print(len(df4_test))
### Generate 17 descriptors ....
df17_train=utilities.generate17(train_set.smiles_canon)
df17_test=utilities.generate17(test_set.smiles_canon)
### Generate 123 descriptors ....'''
df123_train=utilities.generate123(train_set.smiles_canon)
df123_test=utilities.generate123(test_set.smiles_canon)
### Generate 38 feature engineered based on the structure of the smiles ....
df38_train=utilities.generate_features38(train_set.smiles_canon)
df38_test=utilities.generate_features38(test_set.smiles_canon)
### Generate 7 funnctional groups
df7_train=utilities.get_functional_groups(train_set.smiles_canon)
df7_test=utilities.get_functional_groups(test_set.smiles_canon)
### Fingerprint 128....
df128_train=utilities.fingerprint(train_set.smiles_canon,2,128)
df128_test=utilities.fingerprint(test_set.smiles_canon,2,128)


## Prof. Ulf proposed data
df96_train=utilities.generate_desc_96(train_set.smiles_canon)
df96_test=utilities.generate_desc_96(test_set.smiles_canon)

## Prof. Ulf + chatgpt suggested data
df193_train=utilities.generate_desc_193(train_set.smiles_canon)
df193_test=utilities.generate_desc_193(test_set.smiles_canon)

df298_train=pd.concat([df123_train, df128_train, df7_train, df38_train], axis=1)
df298_test=pd.concat([df123_test, df128_test, df7_test, df38_test], axis=1)


print(len(df298_train))
print(len(df298_test))
import os
os.makedirs("cached_features", exist_ok=True)
#Option A: save final combined features
df298_train.to_parquet("offline_data/df298_train.parquet")
df298_test.to_parquet("offline_data/df298_test.parquet")
# Also save masks (needed for GVFA alignment)
# import numpy as np
# np.save("cached_features/train_valid_mask.npy", train_valid)
# np.save("cached_features/test_valid_mask.npy", test_valid)


'''
import os
import numpy as np
import pandas as pd
CACHE = "cached_features"
if os.path.exists(f"{CACHE}/df298_train.parquet"):
    print("Loading cached df298 ...")
    df298_train = pd.read_parquet(f"{CACHE}/df298_train.parquet")
    df298_test  = pd.read_parquet(f"{CACHE}/df298_test.parquet")
    train_valid = np.load(f"{CACHE}/train_valid_mask.npy")
    test_valid  = np.load(f"{CACHE}/test_valid_mask.npy")
    '''
