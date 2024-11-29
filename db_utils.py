import psycopg2
import pandas as pd

class RDSDatabaseConnector:
    def __init__(self, database, user, password, host, port=5432):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                database=self.database,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Database connection successful.")
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            raise

    def fetch_data(self, query):
        if self.connection is None:
            raise Exception("Database connection is not established.")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                data = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(data, columns=columns)
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            raise

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")



