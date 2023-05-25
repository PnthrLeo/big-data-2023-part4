import argparse
import sys
import time

from confluent_kafka import Consumer, KafkaError, KafkaException
from kafka_config_generator import get_kafka_config
from topics import Topic


class CustomConsumer(Consumer):
    def __init__(self, conf: dict, topics: list[str]):
        super().__init__(conf)
        self.cache_messages = None
        self.topics = topics
        
        self.__update_cache(self.topics)

    def __update_cache(self, topics: list[str], timeount: float = 10.0) -> None:
        topic_metadata = self.list_topics(timeout=1.0)
        # check if topics exist
        for topic in topics:
            if topic_metadata.topics.get(topic) is None:
                topics.remove(topic)
        
        messages = []
        
        begin_time = time.time()
        
        try:
            self.subscribe(topics)

            while True:
                current_time = time.time()
                if current_time - begin_time > timeount:
                    self.cache_messages = messages
                    return
                
                msg = self.poll(timeout=1.0)
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                        (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    messages.append(msg)
        finally:
            # Close down consumer to commit final offsets.
            self.close()


    def get_message(self, topic: str, key: str, update_cache: bool = False) -> bytes:
        if self.cache_messages == None or topic not in self.topics or update_cache:
            if topic not in self.topics:
                self.topics.append(topic)
            self.__update_cache(self.topics)
        
        for msg in self.cache_messages:
            if str(msg.key(), 'utf-8') == key and msg.topic() == topic:
                return msg.value()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Kafka Consumer")
    parser.add_argument("--load_model_log", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_log_path", default=None)
    parser.add_argument("--load_model", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--load_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--preds_path", default=None)
    parser.add_argument("--exp_name", required=True)
    args = parser.parse_args()

    config = get_kafka_config()
    topics = [topic.name for topic in Topic]
    
    consumer = CustomConsumer({
        'bootstrap.servers': config['kafka_ip_port'],
        'group.id': config['kafka_consumer_group_id'],
        'enable.auto.commit': False,
        'auto.offset.reset': 'earliest'},
    topics)
    
    if args.load_model_log:
        model_log = consumer.get_message(Topic.ML_LOGS.name, args.exp_name)
        if model_log is not None:
            with open(args.model_log_path, 'wb') as f:
                f.write(model_log)
    
    if args.load_model:
        model = consumer.get_message(Topic.ML_MODELS.name, args.exp_name)
        if model is not None:
            with open(args.model_path, 'wb') as f:
                f.write(model)
    
    if args.load_preds:
        preds = consumer.get_message(Topic.ML_PREDICTS.name, args.exp_name)
        if preds is not None:
            with open(args.preds_path, 'wb') as f:
                f.write(preds)
