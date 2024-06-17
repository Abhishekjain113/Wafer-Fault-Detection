import os 
import sys 
import pickle
import numpy as np
import pandas as pd 
from src.expection import CustomException
from src.logger import logging

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(model.value())[i]
            # training the model
            model.fit(x_train,y_train)
            # predict the test data
            y_test_predict=model.predict(x_test)
            #getting r2 score
            test_model_score=r2_score(y_test,y_test_predict)
            report[list(model.keys())[i]]=test_model_score
        return report
    except Exception as e:
        logging.info("Expecption occured during model traning")
        raise CustomException(e,sys)
    
def load_object(filepath):
    try:
        with open(filepath,'rb') as file_obj:
          return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)