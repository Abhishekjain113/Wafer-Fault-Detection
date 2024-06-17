import os
import sys
from dataclasses import dataclass
from src.expection import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainer()
    def inital_model_trainer(self,train_array,test_array):
        try:
            logging.info('Model training phase')
            #spliting dependent and independent feature
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'XGBClasifier':XGBClassifier(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'RandomForsetClassifier':RandomForestClassifier(),
                'GradientBoosting':GradientBoostingClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier()
            }
            

            logging.info(f"Extracting model config file path")
            models_report: dict=evaluate_model(X_train,Y_train,X_test,Y_test,models)
            # get best model score from dictinoary
            best_model_score=max(sorted(models_report.values()))
            best_model_name=list(models_report.keys())[list(models_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            logging.info("got best model")
            save_object(
                file_path=self.inital_model_trainer.trainer_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("model training faileed")