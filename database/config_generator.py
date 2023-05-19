import os


def get_db_config() -> dict:
    return {
        "db_ip": os.environ["DB_IP"],
        "db_port": int(os.environ["DB_PORT"]),
        "db_name": os.environ["DB_NAME"],
        "db_user": os.environ["DB_USER"],
        "db_password": os.environ["DB_PASSWORD"],
    }
