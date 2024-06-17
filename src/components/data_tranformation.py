import os 
import sys
import numpy as np
import pandas as pd 
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,FunctionTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipleine

from dataclasses import dataclass
from src.logger import logging
from src.expection import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config=DataTransformation()
    
    def get_data_transformation(self):
        try:
            logging.info("Creating data tranformation ")

            # define a custom function for handling na vaue
            replace_na_with_nan =lambda x: np.where(X=='na',np.nan,x)

            #defining preprocessing step 
            nan_replacement_step=('nan_replacement',FunctionTransformer(replace_na_with_nan))
            imputer_step=('imputer',SimpleImputer(strategy='constant',fill_value=0))
            scaler_step=('scaler',RobustScaler())

            #Creating piplines
            preprocessor=Pipleine(
                steps=[
                    nan_replacement_step,
                    imputer_step,
                    scaler_step
                ]
            )
            logging.info("piplines creatation done")
            return preprocessor

        except Exception as e:
            logging.info("data transformation failed")

    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading train ,test datesets
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessor=self.get_data_transformation()
            target_column_name='outcome'
            target_column_mapping={
              '+1':1,
              '-1':0
            }
            # training dataframe
            input_feature_train_df=train_df.drop(columns=['outcome'],axis=1)
            target_feature_train_df=train_df[target_column_name].map(target_column_mapping)
            #testing dataframe
            input_feature_testing_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name].map(target_column_mapping)

            # transforming the dataset
            preprocessor=self.get_data_transformation()

            transformed_input_feature_train_df=preprocessor.fit_transform(input_feature_train_df)
            transformed_input_feature_test_df=preprocessor.transform(input_feature_testing_df)

            # re-sampling of dataset
            smt=SMOTETomek(sampling_strategy='minority')

            input_feature_train_final,target_feature_train_final=smt.fit_transform(transformed_input_feature_train_df,transformed_input_feature_train_df)
            input_feature_test_final,target_feature_test_final=smt.transform(transformed_input_feature_test_df,target_feature_test_df)

            train_arr=np.c_[input_feature_train_final,np.array(target_feature_train_final)]
            test_arr=np.c_[input_feature_test_final,np.array(target_feature_test_final)]

            save_object(self.data_tranformation_config.preprocessor_obj_file_path,
                        obj=preprocessor)
            
            logging.info("Preprocess pickel is creted and saved ")
            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("exception occure in initating data transformation")