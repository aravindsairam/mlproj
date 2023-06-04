import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass                                                                                                          


@dataclass
class DataIngestionConfig:
    data_path = "/home/sai/Documents/end_to_end_ML/mlproj/data/"
    raw_data_path: str = os.path.join(data_path, "StudentsPerformance.csv")
    train_data_path: str = os.path.join(data_path, "train.csv")
    test_data_path: str = os.path.join(data_path, "test.csv")

class DataIngestion(object):
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=2)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Train test data split done")

            return (
                self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
                )
        except Exception as e:
            logging.error("Data ingestion failed")
            raise CustomException(e, sys)
        
    
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
            