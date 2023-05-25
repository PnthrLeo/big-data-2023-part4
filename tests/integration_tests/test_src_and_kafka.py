import sys

sys.path.insert(0, './kafka')
sys.path.insert(0, './src')

import json
import os
import shutil

import pandas as pd
import pytest
from consumer import CustomConsumer
from kafka_config_generator import get_kafka_config
from producer import CustomProducer
from topics import Topic

from classifier import Classifier


@pytest.fixture  
def app_config():
    with open('tests/fixtures/app_config.json') as f:
        config = json.load(f)
    return config


@pytest.fixture  
def kafka_config():
    config = get_kafka_config()
    return config


def test(app_config, kafka_config):
    if not os.path.exists('tests/temp'):
        os.mkdir('tests/temp')
    
    train_path = 'tests/fixtures/train.csv'
    test_path = 'tests/fixtures/test.csv'
    
    pred_path = 'tests/temp/pred.csv'
    new_pred_path = 'tests/temp/new_pred.csv'
    model_path = 'tests/temp/model.pkl'
    new_model_path = 'tests/temp/new_model.pkl'
    log_path = 'tests/temp/log.txt'
    new_log_path = 'tests/temp/new_log.txt'
    
    # 1. Initialize Classifier and train it
    clf = Classifier(train_path=train_path, test_path=test_path, config=app_config)
    
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
    
    # 2. Dump data to files
    pred_df.to_csv(pred_path, index=False)
    clf.save(model_path)
    metrics = {
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1
    }
    with open(log_path, 'w') as f:
        json.dump(metrics, f)
    
    # 3. Initialize Producer and send data to Kafka
    producer = CustomProducer({
        'bootstrap.servers': kafka_config['kafka_ip_port'],
		'client.id': kafka_config['kafka_producer_client_id']
    })
    
    with open(pred_path, 'rb') as f:
        producer.send_message(f.read(), Topic.ML_PREDICTS.name, "test_src_and_kafka")
    with open(model_path, 'rb') as f:
        producer.send_message(f.read(), Topic.ML_MODELS.name, "test_src_and_kafka")
    with open(log_path, 'rb') as f:
        producer.send_message(f.read(), Topic.ML_LOGS.name, "test_src_and_kafka")
    
    # 4. Initialize Consumer and get data from Kafka
    topics = [topic.name for topic in Topic]
    consumer = CustomConsumer({
        'bootstrap.servers': kafka_config['kafka_ip_port'],
        'group.id': kafka_config['kafka_consumer_group_id'],
        'enable.auto.commit': False,
        'auto.offset.reset': 'earliest'},
    topics)
    
    with open(new_pred_path, 'wb') as f:
        f.write(consumer.get_message(Topic.ML_PREDICTS.name, "test_src_and_kafka"))
    with open(new_model_path, 'wb') as f:
        f.write(consumer.get_message(Topic.ML_MODELS.name, "test_src_and_kafka"))
    with open(new_log_path, 'wb') as f:
        f.write(consumer.get_message(Topic.ML_LOGS.name, "test_src_and_kafka"))
    
    # 5. Check that data is the same
    new_pred_df = pd.read_csv(new_pred_path)
    new_clf = Classifier.load(new_model_path)
    with open(new_log_path, 'r') as f:
        new_metrics = json.load(f)
    
    assert(pred_df.equals(new_pred_df))
    assert(metrics == new_metrics)
    
    assert(clf.train_path == new_clf.train_path)
    assert(clf.test_path == new_clf.test_path)
    assert((clf.feature_names == new_clf.feature_names).all())
    assert(clf.config == new_clf.config)
    
    test_df = clf.get_test()
    
    new_test_df = new_clf.get_test()
    _, new_test_f1 = new_clf.predict(new_test_df, 
                                     is_target_provided=True)
    assert(test_f1 == new_test_f1)

    shutil.rmtree('tests/temp')
