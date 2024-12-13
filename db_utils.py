import pandas as pd
import yaml
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

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

class DataTransform:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def convert_columns_to_category(self, columns: list) -> None:
        """
        Convert specified columns from object type to category type.
        """
        for column in columns:
            if self.df[column].dtype == 'object':
                self.df[column] = self.df[column].astype('category')
                print(f"Converted {column} to category.")
            else:
                print(f"{column} is already in category or other type.")

    def convert_columns_to_datetime(self, columns: list) -> None:
        """
        Convert specified columns from object type to datetime type.
        """
        for column in columns:
            if self.df[column].dtype == 'object':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                print(f"Converted {column} to datetime.")
            else:
                print(f"{column} is already in datetime or other type.")

    def convert_columns_to_int64(self, columns: list) -> None:
        """
        Convert specified columns from float64 or object type to int64 type.
        """
        for column in columns:
            if self.df[column].dtype in ['float64', 'object']:
                self.df[column] = self.df[column].fillna(0).astype('int64')
                print(f"Converted {column} to int64.")
            else:
                print(f"{column} is already in int64 or other type.")

class DataFrameInfo:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the class with a DataFrame.
        """
        self.df = df

    def describe_all_columns(self) -> None:
        """
        Describe all columns in the DataFrame to check their data types.
        """
        print("Data types of each column:")
        print(self.df.dtypes)

    def extract_statistical_values(self) -> None:
        """
        Extract and display statistical values such as mean, median, and standard deviation
        for numeric columns only.
        """
        # Filter only numeric columns for the statistics
        numeric_df = self.df.select_dtypes(include=['number'])

        print("\nStatistical Summary of Columns:")

        # Median
        print("Median:")
        print(numeric_df.median())

        # Standard Deviation
        print("\nStandard Deviation:")
        print(numeric_df.std())

        # Mean
        print("\nMean:")
        print(numeric_df.mean())

    
    def count_distinct_values(self) -> None:
        """
        Count distinct values in categorical columns.
        """
        print("\nDistinct Value Counts in Categorical Columns:")
        categorical_columns = self.df.select_dtypes(include='category').columns
        for column in categorical_columns:
            print(f"{column}: {self.df[column].nunique()} distinct values")
    
    def print_shape(self) -> None:
        """
        Print the shape of the DataFrame.
        """
        print("\nShape of the DataFrame:")
        print(self.df.shape)
    
    def count_null_values(self) -> None:
        """
        Generate a count/percentage count of NULL values in each column.
        """
        print("\nCount of NULL Values in Each Column:")
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        null_summary = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentage})
        print(null_summary)

class Plotter:
    """
    A class for visualizing insights from a DataFrame.
    """
    def __init__(self, df):
        """
        Initialize the Plotter class with a DataFrame.
        """
        self.df = df

    def plot_null_values(self, original_df: pd.DataFrame) -> None:
        """
        Visualize the number of NULL values before and after data cleaning.
        """
        # Calculate the number of NULL values in both original and cleaned DataFrames
        original_nulls = original_df.isnull().sum()
        cleaned_nulls = self.df.isnull().sum()

        # Prepare the data for plotting
        null_comparison = pd.DataFrame({
            'Original Nulls': original_nulls,
            'Cleaned Nulls': cleaned_nulls
        })
        null_comparison = null_comparison[null_comparison['Original Nulls'] > 0]  # Filter out columns with no NULLs

        # Plotting the comparison
        plt.figure(figsize=(10, 6))
        null_comparison.plot(kind='bar', stacked=False)
        plt.title("NULL Values Before and After Cleaning")
        plt.ylabel("Count of NULL Values")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    def plot_skewness_distribution(self) -> None:
        """
        Visualize the distribution of skewness values across numeric columns.
        """
        # Calculate skewness for each numeric column
        skewness = self.df.select_dtypes(include=['number']).skew()

        # Plotting the distribution of skewness values
        plt.figure(figsize=(10, 6))
        sns.histplot(skewness, kde=True, bins=20)
        plt.title("Distribution of Skewness Values")
        plt.xlabel("Skewness")
        plt.ylabel("Frequency")
        plt.show()
    
    def get_skewed_columns(self, threshold: float = 1.5) -> dict:
        """
        Get a dictionary of columns with skewness greater than the specified threshold and their skewness values.
        """
        # Calculate skewness for each numeric column
        skewness = self.df.select_dtypes(include=['number']).skew()

        # Get columns with skewness above the threshold
        skewed_columns = skewness[skewness > threshold]
        
        return skewed_columns.to_dict()  # Returns a dictionary with column names and skewness values
    
    def plot_transformed_skew(self, original_df, transformed_df, columns):
        """
        Visualizes the original vs transformed data for skewed columns.
        """
        for col in columns:
            plt.figure(figsize=(12, 6))
            
            # Original Data
            plt.subplot(1, 2, 1)
            sns.histplot(original_df[col], kde=True, bins=20)
            plt.title(f"Original Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            
            # Transformed Data
            plt.subplot(1, 2, 2)
            sns.histplot(transformed_df[col], kde=True, bins=20)
            plt.title(f"Transformed Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            
            plt.tight_layout()
            plt.show()

    def plot_boxplots(self):
        """
        Visualize outliers in the DataFrame using box plots.
        """
        # Select only the numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
    
        # Calculate number of rows and columns for the plot grid
        n_cols = 5 
        n_rows = (len(numeric_cols) // n_cols) + (len(numeric_cols) % n_cols > 0)
    
        # Set up the plot
        plt.figure(figsize=(15, n_rows * 5))
    
        # Create box plots for each numeric column
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)  
            plt.title(f"Box plot for {col}")
            # Plot the box plot for each numeric column
            plt.boxplot(self.df[col].dropna())  # Drop NA values to avoid issues
    
        plt.tight_layout()
        plt.show()

class DataFrameTransform:
    """
    A class for performing EDA transformations on a DataFrame.
    """
    def __init__(self, df):
        """
        Initialize the DataFrameTransform class with a DataFrame.
        
        """
        self.df = df
    
    def drop_columns(self, columns: list) -> None:
        """
        Drop specified columns from the DataFrame.
        """
        try:
            self.df.drop(columns=columns, inplace=True)
            print(f"Successfully dropped columns: {columns}")
        except KeyError as e:
            print(f"Error: {e}. One or more columns not found in DataFrame.")
    
    def impute_with_mode(self, column: str) -> None:
        """
        Impute NULL values in a column with the mode.
        """
        mode_value = self.df[column].mode()[0]  
        self.df[column].fillna(mode_value, inplace=True)
        print(f"Imputed NULL values in '{column}' with mode: {mode_value}")

    def impute_with_median(self, column: str) -> None:
        """
        Impute NULL values in a column with the median value.
        """
        median_value = self.df[column].median() 
        self.df[column].fillna(median_value, inplace=True)
        print(f"Imputed NULL values in '{column}' with median: {median_value}")

    def impute_with_mean(self, column: str) -> None:
        """
        Impute NULL values in a column with the mean value.
        """
        mean_value = self.df[column].mean()  
        self.df[column].fillna(mean_value, inplace=True)
        print(f"Imputed NULL values in '{column}' with mean: {mean_value}")

    def transform_skewed_columns(self, skewed_columns: dict) -> pd.DataFrame:
        """
        Apply transformations (Log, Box-Cox, Yeo-Johnson) to skewed columns 
        and compare the reduction in skewness.
        """
        transformed_df = self.df.copy()
        transformations = ['log', 'boxcox', 'yeojohnson']
        results = {}

        for column, skew_value in skewed_columns.items():
            original_skew = skew_value
            column_data = transformed_df[column]
            column_results = {}

            # Log Transformation 
            if (column_data > 0).all():
                log_transformed = np.log(column_data)
                log_skew = log_transformed.skew()
                column_results['Log Transform'] = log_skew

            # Box-Cox Transformation
            if (column_data > 0).all():
                boxcox_transformed, _ = stats.boxcox(column_data)
                boxcox_skew = pd.Series(boxcox_transformed).skew()
                column_results['Box-Cox Transform'] = boxcox_skew

            # Yeo-Johnson Transformation 
            yeo_johnson_transformed, _ = stats.yeojohnson(column_data)
            yeo_johnson_skew = pd.Series(yeo_johnson_transformed).skew()
            column_results['Yeo-Johnson Transform'] = yeo_johnson_skew

            results[column] = {
                'Original Skew': original_skew,
                'Log Transform Skew': column_results.get('Log Transform', None),
                'Box-Cox Transform Skew': column_results.get('Box-Cox Transform', None),
                'Yeo-Johnson Transform Skew': column_results['Yeo-Johnson Transform'],
            }

            # Apply the transformation that results in the smallest skew
            best_transform = min(column_results, key=column_results.get)
            if best_transform == 'Log Transform' and 'Log Transform' in column_results:
                transformed_df[column] = np.log(column_data)
            elif best_transform == 'Box-Cox Transform' and 'Box-Cox Transform' in column_results:
                transformed_df[column] = stats.boxcox(column_data)[0]
            elif best_transform == 'Yeo-Johnson Transform':
                transformed_df[column] = stats.yeojohnson(column_data)[0]

        # Print the results of the skewness comparison
        for column, transformation_results in results.items():
            print(f"Skewness comparison for {column}:")
            for transform, skew in transformation_results.items():
                print(f"{transform}: {skew}")
            print("-" * 50)

        return transformed_df

    def remove_outliers(self):
        """
        Remove outliers from the DataFrame using the IQR method.
        """
        # Select only numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
        Q1 = self.df[numeric_cols].quantile(0.25)
        Q3 = self.df[numeric_cols].quantile(0.75)
        
        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1
        
        # Calculate the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the data by removing outliers outside the bounds
        self.df = self.df[~((self.df[numeric_cols] < lower_bound) | (self.df[numeric_cols] > upper_bound)).any(axis=1)]
        
        return self.df
    
    
 