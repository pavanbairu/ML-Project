from dataclasses import dataclass
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import FraudException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r"D:\Data Science\github\Projects\ML\ML-Project\Notebooks\dataset\stud.csv")
            logging.info("read the dataset as dataframe df")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            logging.info("split train and test data")

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False, header=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path)
        except Exception as e:
            raise FraudException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()