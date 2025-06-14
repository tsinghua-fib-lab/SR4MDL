import logging
import numpy as np
import nd2py as nd
import pandas as pd
from typing import Dict


logger = logging.getLogger(__name__)


def preprocess(X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray]):
    # Dictize X
    if isinstance(X, np.ndarray):
        X = {f'x_{i+1}': X[:, i] for i in range(X.shape[1])} # (N, D) -> {x_1: (N,), x_2: (N,), ...}
    elif isinstance(X, pd.DataFrame):
        X = {col: X[col].values for col in X.columns}
    elif isinstance(X, dict):
        pass
    else:
        raise ValueError(f'Unknown type: {type(X)}')
    return X


def sample_Xy(X:Dict[str,np.ndarray], y:np.ndarray, sample_num):
    if len(y) > sample_num:
        logger.info(f'Randomly sample {sample_num} samples from {len(y)} samples')
        idx = np.random.choice(len(y), sample_num, replace=False)
        X = {k: v[idx] for k, v in X.items()}
        y = y[idx]
    return X, y


def rename_variable(eqtree:nd.Symbol, variable_mapping:Dict[str,str]):
    eqtree = eqtree.copy()
    for node in eqtree.iter_postorder():
        if isinstance(node, nd.Variable):
            node.name = variable_mapping.get(node.name, node.name)
    return eqtree
