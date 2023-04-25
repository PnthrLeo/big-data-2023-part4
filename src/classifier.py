# Modified from: https://github.com/Danielto1404/mle-template/blob/main/src/train.py
import json
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import argparse

from typing import Tuple
from utils import preprocess, get_X_y, split_for_validation
import os


with open("config.json") as f:
    config = json.load(f)


class Classifier:
    def __init__(self, train_path: str, test_path: str):
        self.model = self.init_model()
        self.train_path = train_path
        self.test_path = test_path
        self.feature_names = None
    
    def init_model(self):
        supported_models = ["LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier"]
        if config['model']['name'] not in supported_models:
            raise ValueError(f"Model {config['model']['name']} is not supported")

        elif config['model']['name'] == "RandomForestClassifier":
            return RandomForestClassifier(**config['model']['hyperparameters'])
        elif config['model']['name'] == "DecisionTreeClassifier":
            return DecisionTreeClassifier(**config['model']['hyperparameters'])

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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts on data
        :param df: DataFrame - data to predict on
        :return: DataFrame - predictions
        """
        df = preprocess(df)
        df = df[self.feature_names]
        return self.model.predict(df)

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
    parser.add_argument("--from_ckpt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--exp_name", required=True)
    args = parser.parse_args()
    
    logging.basicConfig(filename=f"experiments/{args.exp_name}/run.log", encoding='utf-8', level=logging.DEBUG)
    
    if args.train is None and args.test is None:
        raise ValueError("Train and test paths are not provided")

    if args.from_ckpt and args.train is None:
        classifier = Classifier.load(args.ckpt_path)
        classifier.train_path = args.train
        classifier.test_path = args.test
    elif not args.from_ckpt and args.train is not None:
        classifier = Classifier(args.train, args.test)
        train_f1, valid_f1 = classifier.fit(use_validation=True)

        logging.info(f"Train F1 {train_f1} | Valid F1 {valid_f1}")
        os.makedirs(f"experiments/{args.exp_name}", exist_ok=True)
        model_save_path = f"experiments/{args.exp_name}/model.pkl"
        classifier.save(model_save_path)
    
    if args.test is not None:
        labels = classifier.predict(classifier.get_test())
        test_preds_path = f"experiments/{args.exp_name}/test_preds.csv"
        pd.DataFrame({"label": labels}).to_csv(test_preds_path, header=True, index=False)
