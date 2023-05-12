import argparse
import json
import logging
import time

import pandas as pd
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile
from cassandra.policies import WhiteListRoundRobinPolicy


class Database:
    def __init__(self, config: dict):
        profile = ExecutionProfile(load_balancing_policy=WhiteListRoundRobinPolicy(['0.0.0.0']))
        
        try:
            ap = PlainTextAuthProvider(username='cassandra', password='cassandra')
            cluster = Cluster(port=9042, auth_provider=ap, execution_profiles={'default': profile})
            session = cluster.connect()

            session.execute(f"""CREATE ROLE IF NOT EXISTS {config['db_user']}
                            WITH SUPERUSER = true AND
                            LOGIN = true AND
                            PASSWORD = '{config['db_password']}'""")
            session.shutdown()
        except:
            logging.warning("cassandra role does not exist")

        ap = PlainTextAuthProvider(
            username=config['db_user'], password=config['db_password'])
        cluster = Cluster(port=9042, auth_provider=ap, execution_profiles={'default': profile})
        session = cluster.connect()

        session.execute("""DROP ROLE IF EXISTS cassandra""")
        session.execute(f"CREATE KEYSPACE IF NOT EXISTS {config['db_name']} WITH REPLICATION =" +
                        "{ 'class' : 'SimpleStrategy', 'replication_factor' : 2 }")

        self.session = session
        self.db_name = config['db_name']

    def save_to_db(self, csv_path: str, table_name: str):
        list_of_tables = self.session.execute(f"""SELECT table_name FROM system_schema.tables
                                            WHERE keyspace_name = '{self.db_name}'""")
        list_of_tables = [row[0] for row in list_of_tables.all()]
        if table_name in list_of_tables:
            raise ValueError(f"Table {table_name} already exists")

        self.session.execute(f"""CREATE TABLE IF NOT EXISTS {self.db_name}.{table_name} (
                             id int,
                             area float,
                             perimeter float,
                             compactness float,
                             kernel_length float,
                             kernel_width float,
                             asymmetry_coeff float,
                             kernel_groove float,
                             type int,
                             PRIMARY KEY (id))""")
        time.sleep(5)
        
        df = pd.read_csv(csv_path)
        
        if 'Type' not in df.columns:
            df['Type'] = 0
        
        for i in range(len(df)):
            self.session.execute(f"INSERT INTO {self.db_name}.{table_name} (id, area, perimeter, compactness, kernel_length," + 
                                 "kernel_width, asymmetry_coeff, kernel_groove, type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                 (i, df.iloc[i]['Area'], df.iloc[i]['Perimeter'], df.iloc[i]['Compactness'], df.iloc[i]['Kernel.Length'],
                                  df.iloc[i]['Kernel.Width'], df.iloc[i]['Asymmetry.Coeff'], df.iloc[i]['Kernel.Groove'], int(df.iloc[i]['Type'])))

    def load_from_db(self, csv_path: str, table_name: str):
        df = pd.DataFrame(list(self.session.execute(f"""SELECT * FROM {self.db_name}.{table_name}""")))
        df = df[['id', 'area', 'perimeter', 'compactness', 'kernel_length', 
                 'kernel_width', 'asymmetry_coeff', 'kernel_groove', 'type']]

        features_list = ['Area', 'Perimeter', 'Compactness', 'Kernel.Length',
                         'Kernel.Width', 'Asymmetry.Coeff', 'Kernel.Groove']
        df.columns = ['id'] + features_list + ['Type']
        df[features_list] = df[features_list].apply(lambda x: round(x, 4))
        
        df = df.sort_values(by=['id'])
        df.drop(columns=['id'], inplace=True)
        df.to_csv(csv_path, index=False)
    
    def del_from_db(self, table_name: str):
        self.session.execute(f"""DROP TABLE IF EXISTS {self.db_name}.{table_name}""")
        time.sleep(5)
    
    def __del__(self):
        self.session.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Database")
    parser.add_argument("--delete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delete_table_name", default=None)
    parser.add_argument("--save_table_name", default=None)
    parser.add_argument("--save_csv_path", default=None)
    parser.add_argument("--load_table_name", default=None)
    parser.add_argument("--load_csv_path", default=None)
    args = parser.parse_args()
    
    with open("database/config.json") as f:
        config = json.load(f)
    
    db = Database(config)
    if args.delete:
        db.del_from_db(args.delete_table_name)
    if args.save:
        db.save_to_db(args.save_csv_path, args.save_table_name)
    if args.load:
        db.load_from_db(args.load_csv_path, args.load_table_name)
