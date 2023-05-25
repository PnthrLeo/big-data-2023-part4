import argparse
from typing import Union

from confluent_kafka import Producer
from kafka_config_generator import get_kafka_config
from topics import Topic


class CustomProducer(Producer):
    def __init__(self, conf: dict):
        super().__init__(conf)
    
    def send_message(self, msg: Union[str, bytes, int, float], topic: str, key: str):
        self.produce(topic, key=key, value=msg)
        self.flush()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Kafka Producer")
    parser.add_argument("--save_model_log", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_log_path", default=None)
    parser.add_argument("--save_model", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--save_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--preds_path", default=None)
    parser.add_argument("--exp_name", required=True)
    args = parser.parse_args()
    
    config = get_kafka_config()
    
    producer = CustomProducer({
        'bootstrap.servers': config['kafka_ip_port'],
		'client.id': config['kafka_producer_client_id']
    })
    
    if args.save_model_log:
        with open(args.model_log_path, 'rb') as f:
            producer.send_message(f.read(), Topic.ML_LOGS.name, args.exp_name)
    
    if args.save_model:
        with open(args.model_path, 'rb') as f:
            producer.send_message(f.read(), Topic.ML_MODELS.name, args.exp_name)
    
    if args.save_preds:
        with open(args.preds_path, 'rb') as f:
            producer.send_message(f.read(), Topic.ML_PREDICTS.name, args.exp_name)
