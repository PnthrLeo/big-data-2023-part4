import sys

sys.path.insert(0, './src')

import json
import os

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier import Classifier
from utils import get_X_y


@pytest.fixture  
def config():
    with open("tests/fixtures/app_config.json") as f:
        config = json.load(f)
    return config


def test_init_model(config):
    config['model']['name'] = 'RandomForestClassifier'
    config['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 2,
            'random_state': 42
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    assert(type(clf.model) == RandomForestClassifier)
    
    config['model']['name'] = 'DecisionTreeClassifier'
    config['model']['hyperparameters'] = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': 100
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    assert(type(clf.model) == DecisionTreeClassifier)
    

def test_get_train_test(config):
    config['model']['name'] = 'RandomForestClassifier'
    config['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 2,
            'random_state': 42
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    
    train_df = clf.get_train()
    test_df = clf.get_test()
    
    true_train_df = pd.read_csv(train_path)
    true_test_df = pd.read_csv(test_path)
    
    assert(type(train_df) == pd.DataFrame)
    assert(type(test_df) == pd.DataFrame)
    assert(train_df.equals(true_train_df))
    assert(test_df.equals(true_test_df))


def test_fit(config):
    config['model']['name'] = 'RandomForestClassifier'
    config['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 2,
            'random_state': 42
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    train_f1, val_f1 = clf.fit(use_validation=False)
    assert(train_f1 == 0.9321896874691905)
    assert(val_f1 == None)
    
    train_f1, val_f1 = clf.fit(use_validation=True)
    assert(train_f1 == 0.9454258250991464)
    assert(val_f1 == 0.7842650103519668)


def test_predict(config):
    config['model']['name'] = 'RandomForestClassifier'
    config['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 2,
            'random_state': 42
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    clf.fit(use_validation=False)
    
    mini_test_df = clf.get_test()[:3]
    y_pred, _ = clf.predict(mini_test_df)
    assert((y_pred == [2, 1, 2]).all())


def test_evaluate(config):
    config['model']['name'] = 'RandomForestClassifier'
    config['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 2,
            'random_state': 42
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    clf.fit(use_validation=False)
    
    test_df = clf.get_test()
    X, y = get_X_y(test_df)
    test_f1 = clf.evaluate(X, y)
    assert(test_f1 == 0.8706206206206207)


def test_save_load(config):
    config['model']['name'] = 'RandomForestClassifier'
    config['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 2,
            'random_state': 42
    }
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    model_save_path = './experiments/unit_test/model.pkl'
    os.makedirs(f"./experiments/unit_test", exist_ok=True)
    
    clf = Classifier(train_path=train_path, test_path=test_path, config=config)
    clf.fit(use_validation=False)
    clf.save(model_save_path)
    
    clf2 = Classifier.load(model_save_path)
    assert(clf.train_path == clf2.train_path)
    assert(clf.test_path == clf2.test_path)
    assert((clf.feature_names == clf2.feature_names).all())
    assert(clf.config == clf2.config)
    
    test_df = clf.get_test()
    X, y = get_X_y(test_df)
    test_f1_clf = clf.evaluate(X, y)
    test_f1_clf2 = clf2.evaluate(X, y)
    
    assert(test_f1_clf == test_f1_clf2)
