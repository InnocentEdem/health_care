import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException

def load_dataset(dataset_name, download_path="data"):
    """
    Downloads and loads a dataset from Kaggle using the Kaggle API.
    
    Args:
        dataset_name (str): The name of the dataset in the format "user/dataset".
        download_path (str): The directory where the dataset should be downloaded.
    
    Returns:
        pd.DataFrame or None: The loaded dataset as a Pandas DataFrame, or None if an error occurred.
    """
    # Create the download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Initialize the Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
    except ApiException as e:
        print(f"Kaggle API authentication failed: {e}")
        return None

    try:
        # Download the dataset
        print(f"Downloading dataset: {dataset_name} to '{download_path}'")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        # Identify CSV files
        csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
        if not csv_files:
            print("No CSV files found in the downloaded dataset.")
            return None
        
        # Handle multiple CSVs if they exist
        if len(csv_files) > 1:
            print(f"Multiple CSV files found: {csv_files}")
            print("Loading the first CSV by default.")
        
        csv_file_path = os.path.join(download_path, csv_files[0])
        print(f"Loading file: {csv_file_path}")
        
        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(csv_file_path)
        print("Dataset loaded successfully.")
        return df

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ApiException as e:
        print(f"Kaggle API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None
