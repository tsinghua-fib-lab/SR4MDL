import numpy as np
import pandas as pd
from typing import Dict
from ..env import Symbol, Variable


def preprocess_X(X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray]):
    if isinstance(X, np.ndarray):
        variable_mapping = {f'x_{i+1}': f'x_{i+1}' for i in range(X.shape[1])}
        X = {f'x_{i+1}': X[:, i] for i in range(X.shape[1])}
    elif isinstance(X, pd.DataFrame):
        variable_mapping = {f'x_{i+1}': col for i, col in enumerate(X.columns)}
        X = {f'x_{i+1}': X[col].values for i, col in enumerate(X.columns)}
    elif isinstance(X, dict):
        variable_mapping = {f'x_{i+1}': col for i, col in enumerate(X.keys())}
        X = {f'x_{i+1}': X[col] for i, col in enumerate(X.keys())}
    else:
        raise ValueError(f'Unknown type: {type(X)}')
    return X, variable_mapping


def rename_variable(eqtree:Symbol, variable_mapping:Dict[str,str]):
    eqtree = eqtree.copy()
    for node in eqtree.postorder():
        if isinstance(node, Variable):
            node.name = variable_mapping.get(node.name, node.name)
    return eqtree
