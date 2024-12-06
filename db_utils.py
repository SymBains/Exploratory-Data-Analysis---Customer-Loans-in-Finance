import pandas as pd
import yaml
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    """
    A class containing methods to interact with an RDS database and manage data operations.
    """

    def __init__(self, credentials: dict):
        """
        Initialize the RDSDatabaseConnector with database credentials.
        """
        self.host = credentials['RDS_HOST']
        self.user = credentials['RDS_USER']
        self.password = credentials['RDS_PASSWORD']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']
        self.engine = None

    def initialize_engine(self) -> None:
        """
        Initialize a SQLAlchemy engine for database connections.
        """
        try:
            connection_string = (
                f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            )
            self.engine = create_engine(connection_string)
            print("SQLAlchemy engine initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLAlchemy engine: {e}")

    def fetch_data(self, table_name: str = 'loan_payments') -> pd.DataFrame:
        """
        Fetch data from a specified table in the RDS database.
        """
        if self.engine is None:
            raise ValueError("Engine is not initialized. Call initialize_engine() first.")
        try:
            query = f"SELECT * FROM {table_name};"
            return pd.read_sql(query, self.engine)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from table {table_name}: {e}")

    def close_connection(self) -> None:
        """
        Close the database connection.
        """
        if self.engine:
            self.engine.dispose()
            print("Database connection closed.")

    @staticmethod
    def load_credentials(file_path: str = 'credentials.yaml') -> dict:
        """
        Load database credentials from a YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Credentials file not found at {file_path}.")
        except Exception as e:
            raise RuntimeError(f"Error occurred while loading credentials: {e}")

    @staticmethod
    def save_data_to_csv(df: pd.DataFrame, file_path: str = 'loan_payments.csv') -> None:
        """
        Save a Pandas DataFrame to a CSV file.
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}.")
        except Exception as e:
            raise RuntimeError(f"Error occurred while saving data to CSV: {e}")

    @staticmethod
    def load_data_from_csv(file_path: str = 'loan_payments.csv') -> pd.DataFrame:
        """
        Load data from a CSV file into a Pandas DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}.")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {file_path} was not found.")
        except Exception as e:
            raise RuntimeError(f"Error occurred while loading data: {e}")


if __name__ == "__main__":
    # Load credentials
    try:
        credentials = RDSDatabaseConnector.load_credentials()
        connector = RDSDatabaseConnector(credentials)

        # Initialize engine and fetch data
        connector.initialize_engine()
        df = connector.fetch_data('loan_payments')

        # Save and reload data
        connector.save_data_to_csv(df, 'loan_payments.csv')
        df = RDSDatabaseConnector.load_data_from_csv('loan_payments.csv')

        # Display data summary
        print(df.info())
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            connector.close_connection()
        except NameError:
            pass  
