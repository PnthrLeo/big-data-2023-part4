import sys

sys.path.insert(0, './kafka')

import socket

import pytest
from consumer import CustomConsumer
from producer import CustomProducer
from kafka_config_generator import get_kafka_config
from topics import Topic


def test_consumer_producer():
    msg = "Hello World!"
    
    config = get_kafka_config()
    topics = [topic.name for topic in Topic]
    
    producer = CustomProducer({
        'bootstrap.servers': config['kafka_ip_port'],
		'client.id': config['kafka_producer_client_id']
    })
    producer.send_message(msg, Topic.ML_LOGS.name, "test")
    
    consumer = CustomConsumer({
        'bootstrap.servers': config['kafka_ip_port'],
        'group.id': config['kafka_consumer_group_id'],
        'enable.auto.commit': False,
        'auto.offset.reset': 'earliest'},
    topics)
    recieved_msg = consumer.get_message(Topic.ML_LOGS.name, "test")
    recieved_msg = str(recieved_msg, 'utf-8')

    assert msg == recieved_msg
