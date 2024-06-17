import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.expection import CustomException

# Initialize data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

# Creating a data ingestion class
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            # Read the dataset
            df = pd.read_csv(os.path.join('notebooks/data', 'Inputfile.csv'))
            logging.info('Data Ingestion starts')

            # Ensure directory path exists and save the raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train test splitting of dataset starting")
            train, test = train_test_split(df, test_size=0.3, random_state=42)

            # Ensure directory path exists and save the train and test dataset
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train.to_csv(self.ingestion_config.train_data_path, index=False)

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train test split done")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print(f"Train Data Path: {train_data_path}")
    print(f"Test Data Path: {test_data_path}")
