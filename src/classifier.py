# Modified from: https://github.com/Danielto1404/mle-template/blob/main/src/train.py
import argparse
import json
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from utils import get_X_y, preprocess, split_for_validation


class Classifier:
    def __init__(self, train_path: str, test_path: str, config: dict):
        self.train_path = train_path
        self.test_path = test_path
        self.config = config
        self.feature_names = None
        self.model = self.init_model()
    
    def init_model(self):
        supported_models = ["RandomForestClassifier", "DecisionTreeClassifier"]
        if self.config['model']['name'] not in supported_models:
            raise ValueError(f"Model {self.config['model']['name']} is not supported")

        elif self.config['model']['name'] == "RandomForestClassifier":
            return RandomForestClassifier(**self.config['model']['hyperparameters'])
        elif self.config['model']['name'] == "DecisionTreeClassifier":
            return DecisionTreeClassifier(**self.config['model']['hyperparameters'])

    def get_train(self) -> pd.DataFrame:
        """Returns train data as pandas DataFrame
        """
        return pd.read_csv(self.train_path)

    def get_test(self) -> pd.DataFrame:
        """Returns test data as pandas DataFrame"""
        return pd.read_csv(self.test_path)

    def fit(self, use_validation: bool = False) -> Tuple[float, float]:
        """
        Fits model on train data
        :param use_validation: bool - whether to use validation data
        :return: train_f1, val_f1 - f1 scores for train and validation data
        """
        train_df = self.get_train()
        train_df = preprocess(train_df)
        X_train, y_train = get_X_y(train_df)
        self.feature_names = X_train.columns

        if use_validation:
            X_train, X_val, y_train, y_val = split_for_validation(X_train, y_train)
            self.model.fit(X_train, y_train)
            train_f1 = self.evaluate(X_train, y_train)
            val_f1 = self.evaluate(X_val, y_val)
            return train_f1, val_f1
        else:
            self.model.fit(X_train, y_train)
            train_f1 = self.evaluate(X_train, y_train)
            return train_f1, None

    def evaluate(self, X: pd.Series, y: pd.Series) -> float:
        """
        Evaluates model on x and y
        :param x: Series - x data
        :param y: Series - y data
        :return: macro f1 score
        """
        y_pred = self.model.predict(X)
        return f1_score(y, y_pred, average="macro")

    def predict(self, df: pd.DataFrame, is_target_provided: bool = False) -> np.ndarray:
        """
        Predicts on data
        :param df: DataFrame - data to predict on
        :param is_target_provided: bool - whether target is provided
        :return: DataFrame, float - predictions, f1 score
        """
        if not is_target_provided:
            df = preprocess(df)
            df = df[self.feature_names]
            return self.model.predict(df), None
        else:
            df = preprocess(df)
            X, y = get_X_y(df)
            X = X[self.feature_names]
            return self.model.predict(X), self.evaluate(X, y)

    def save(self, cls_path: str):
        with open(cls_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, cls_path: str):
        with open(cls_path, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Classifier")
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--predict", default=None)
    parser.add_argument("--from_ckpt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--exp_name", required=True)
    args = parser.parse_args()
    
    logging.basicConfig(filename=f"experiments/{args.exp_name}/run.log", encoding='utf-8', level=logging.DEBUG)
    
    if args.train is None and args.test is None and args.predict is None:
        raise ValueError("Train and test paths are not provided")
    
    if args.from_ckpt and args.train is not None:
        raise ValueError("Train path is provided but from_ckpt is True")
    
    with open("config.json") as f:
        config = json.load(f)

    if args.from_ckpt:
        classifier = Classifier.load(args.ckpt_path)
        classifier.test_path = args.test
    elif not args.from_ckpt and args.train is not None:
        classifier = Classifier(args.train, args.test, config)
        train_f1, valid_f1 = classifier.fit(use_validation=True)

        logging.info(f"Train F1 {train_f1} | Valid F1 {valid_f1}")
        os.makedirs(f"experiments/{args.exp_name}", exist_ok=True)
        model_save_path = f"experiments/{args.exp_name}/model.pkl"
        classifier.save(model_save_path)
    
    if args.test is not None:
        test_df = classifier.get_test()
        test_df['Type'], test_f1 = classifier.predict(test_df, is_target_provided=True)
        logging.info(f"Test F1 {test_f1}")
        test_preds_path = f"experiments/{args.exp_name}/test_preds.csv"
        test_df.to_csv(test_preds_path, header=True, index=False)
    
    if args.predict is not None:
        predict_df = pd.read_csv(args.predict)
        predict_df['Type'], _ = classifier.predict(predict_df, is_target_provided=False)
        predict_preds_path = f"experiments/{args.exp_name}/predict_preds.csv"
        predict_df.to_csv(predict_preds_path, header=True, index=False)
