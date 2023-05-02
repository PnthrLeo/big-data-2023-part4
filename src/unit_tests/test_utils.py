import sys

sys.path.insert(0, './src')

import json
import random

import numpy as np
import pandas as pd
import pytest

from utils import (dropna, get_X_y, preprocess, set_right_dtypes,
                   split_for_validation)

with open("config.json") as f:
    config = json.load(f)
    

@pytest.fixture    
def df() -> pd.DataFrame:
    col_names = config['data']['dtypes']
    df = pd.DataFrame(columns=col_names.keys())
    
    for i in range(10):
        df.loc[i] = [np.random.randint(0, 100) for _ in range(len(col_names))]
    
    for col in df.columns:
        if random.random() < 0.5:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
    
    return df


def test_dropna():
    df = pd.DataFrame({'a': [1, 2, 3, np.nan, 5], 'b': [6, 7, np.nan, 8, 9]})
    df = dropna(df)
    assert df.shape == (3, 2)
    assert df.values.sum() == 30


def test_set_right_dtypes(df: pd.DataFrame):
    df = set_right_dtypes(df)
    for col in df.columns:
        assert df[col].dtype == config['data']['dtypes'][col]


def test_preprocess(df: pd.DataFrame):
    shape = df.shape
    sum_of_values = df.values.sum()
    df = preprocess(df)
    assert df.shape == shape
    assert df.values.sum() == sum_of_values


def test_get_X_y(df: pd.DataFrame):
    row_num, col_num = df.shape
    X, y = get_X_y(df)
    assert type(X) == pd.DataFrame
    assert type(y) == pd.Series
    
    if not config['data']['feature_names']:
        assert X.shape == (row_num, col_num - 1)
    else:
        assert X.shape == (row_num, len(config['data']['feature_names']))
    assert y.shape == (row_num,)


def test_split_for_validation(df: pd.DataFrame):
    X, y = get_X_y(df)
    X_train, X_val, y_train, y_val = split_for_validation(X, y)
    assert type(X_train) == pd.DataFrame
    assert type(X_val) == pd.DataFrame
    assert type(y_train) == pd.Series
    assert type(y_val) == pd.Series
    assert X_train.shape == (int(0.8 * X.shape[0]), X.shape[1])
    assert X_val.shape == (int(0.2 * X.shape[0]), X.shape[1])
    assert y_train.shape == (int(0.8 * y.shape[0]),)
    assert y_val.shape == (int(0.2 * y.shape[0]),)
    