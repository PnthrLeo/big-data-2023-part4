import sys

sys.path.insert(0, './database')
sys.path.insert(0, './src')

import json
import os
import shutil

import pandas as pd
import pytest

from classifier import Classifier
from database_config_generator import get_db_config
from database import Database


@pytest.fixture  
def app_config():
    with open('tests/fixtures/app_config.json') as f:
        config = json.load(f)
    return config


@pytest.fixture  
def db_config():
    config = get_db_config()
    return config


def test(app_config, db_config):
    if not os.path.exists('tests/temp'):
        os.mkdir('tests/temp')
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    
    new_train_path = 'tests/temp/train.csv'
    new_test_path = 'tests/temp/test.csv'
    pred_path = 'tests/temp/pred.csv'
    new_pred_path = 'tests/temp/new_pred.csv'
    
    
    save_data_to_db(db_config, train_path, test_path)
    load_data_from_db(db_config, new_train_path, new_test_path)
    
    clf = Classifier(train_path=new_train_path, test_path=new_test_path, config=app_config)
    
    train_f1, val_f1 = clf.fit(use_validation=True)
    assert(train_f1 == 0.9454258250991464)
    assert(val_f1 == 0.7842650103519668)
    
    test_df = clf.get_test()
    pred_df = test_df.copy()
    preds, test_f1 = clf.predict(pred_df, 
                                 is_target_provided=True)
    pred_df['Type'] = None
    pred_df['Type'] = preds
    assert(test_f1 == 0.7973395026026605)
    
    pred_df.to_csv(pred_path, index=False)
    save_pred_to_db(db_config, pred_path)
    
    load_pred_from_db(db_config, new_pred_path)
    new_pred_df = pd.read_csv(new_pred_path)
    assert(pred_df.equals(new_pred_df))
    
    del_tables(db_config)
    shutil.rmtree('tests/temp')

def save_data_to_db(db_config, train_path, test_path):
    db = Database(db_config)
    db.save_to_db(train_path, 'train')  
    db.save_to_db(test_path, 'test')


def load_data_from_db(db_config, train_path, test_path):
    db = Database(db_config)
    db.load_from_db(train_path, 'train')  
    db.load_from_db(test_path, 'test')


def save_pred_to_db(db_config, pred_path):
    db = Database(db_config)
    db.del_from_db('pred')
    db.save_to_db(pred_path, 'pred')


def load_pred_from_db(db_config, pred_path):
    db = Database(db_config)
    db.load_from_db(pred_path, 'pred')
    

def del_tables(db_config):
    db = Database(db_config)
    db.del_from_db('train')
    db.del_from_db('test')
    db.del_from_db('pred')
