import os
import sys


from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    object=DataIngestion()
    train_data_path,test_data_path=object.initiste_data_ingestion()

    datatransformation=DataTransformation()
    train_arr,test_arr,_=datatransformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.inital_model_trainer(train_arr,test_arr)
    
