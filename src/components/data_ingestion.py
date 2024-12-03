import pandas as pd
import os
import sys
from ..logger import logging
from ..exception import CustomException
from src.components.data_transformation_training import MainTransform
from src.components.data_transformation_training import datatransformation
from sklearn.model_selection import train_test_split



class data_ingest():
    def __init__(self):
        self.raw_data_path = os.path.join('archieve','raw.csv')
        self.train_data_path = os.path.join('archieve','train_data.csv')
        self.test_data_path = os.path.join('archieve','test_data.csv')

class IngestCall():
    def __init__(self):
        self.ingestion = data_ingest()
    
    def InitiatinIngestion(self):
        try:
            
            logging.info("Starting data ingestion process.")

            df = pd.read_csv(r'notebook/data/df_train')
        
            logging.info("Data files read successfully.")

            df = df.drop(columns=['ArticleId'], axis=1)
        
            logging.info("'ArticleId' column dropped from training data.")

            df.to_csv(self.ingestion.raw_data_path, header=True, index=False)

            logging.info("Raw data is stored successfully")

            df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

            logging.info("Data Split is completed")

            logging.info(f"Training data saved to {self.ingestion.train_data_path}.")

            os.makedirs(os.path.dirname(self.ingestion.raw_data_path), exist_ok=True)

            df_train.to_csv(self.ingestion.train_data_path, header=True,index=False)

            df_test.to_csv(self.ingestion.test_data_path, header=True,index=False)
        
            logging.info(f"Test data saved to {self.ingestion.test_data_path}.")


            return self.ingestion.raw_data_path, self.ingestion.train_data_path
        
        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise CustomException(e,sys)


if __name__ =="__main__":
    logging.info("Data ingestion script started.")

    obj = IngestCall()
    train_data, test_data = obj.InitiatinIngestion()
    logging.info("Data ingestion process completed.")
        
    data_transoformation  = MainTransform()
    train_arr,test_arr,preprocessor= data_transoformation.data_transform_initiate(train_data,test_data)


