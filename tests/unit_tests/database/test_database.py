import sys

sys.path.insert(0, './database')

import os

import pandas as pd
import pytest

from config_generator import get_db_config
from database import Database


@pytest.fixture  
def config():
    config = get_db_config()
    return config


def get_list_of_tables(db: Database) -> list[str]:
    list_of_tables = db.session.execute(f"""SELECT table_name FROM system_schema.tables
                                        WHERE keyspace_name = '{db.db_name}'""")
    list_of_tables = [row[0] for row in list_of_tables.all()]
    return list_of_tables


def test_del(config):
    db = Database(config)

    db.save_to_db('tests/fixtures/train.csv', 'train')  
    assert('train' in get_list_of_tables(db))
    
    db.del_from_db('train')
    assert('train' not in get_list_of_tables(db))


def test_save_load(config):
    db = Database(config)
    
    db.save_to_db('tests/fixtures/train.csv', 'train')
    db.load_from_db('tests/fixtures/new_train.csv', 'train')
    df = pd.read_csv('tests/fixtures/train.csv')
    new_df = pd.read_csv('tests/fixtures/new_train.csv')

    db.del_from_db('train')
    os.remove('tests/fixtures/new_train.csv')
    
    assert(df.equals(new_df))
