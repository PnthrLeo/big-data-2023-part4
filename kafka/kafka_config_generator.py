import os
import socket


def get_kafka_config() -> dict:
    return {
        "kafka_ip_port": os.environ["KAFKA_IP_PORT"],
        "kafka_consumer_group_id": os.environ["KAFKA_CONSUMER_GROUP_ID"],
        "kafka_producer_client_id": socket.gethostname()
    }
