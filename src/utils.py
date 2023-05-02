import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open("config.json") as f:
    config = json.load(f)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = dropna(df)
    df = set_right_dtypes(df)
    return df

def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[config['data']['target_name']]
    
    if not config['data']['feature_names']:
        X = df.drop(config['data']['target_name'], axis=1)
    else:
        X = df[config['data']['feature_names']]
    
    return X, y

def split_for_validation(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def dropna(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def set_right_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    dict_dtypes = config['data']['dtypes']
    
    for column in df.columns:
        if column in dict_dtypes.keys():
            df[column] = df[column].astype(dict_dtypes[column])
        else:
            ValueError(f"Column {column} is not in config file")
    
    return df
