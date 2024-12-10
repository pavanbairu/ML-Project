import os
import sys

from src.exception import FraudException
from src.logger import logging
from src.utils import save_obj

import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocessing_obj_path = os.path.join("artifacts", "preprocessing.pkl")


class DataTransformation():

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):

        try:

            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course",
                ]
            
            num_pieline = Pipeline(
                steps=[

                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessing_obj = ColumnTransformer(
                [
                    ("num_pipeline", num_pieline, numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessing_obj
        
        except Exception as e:
            raise FraudException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_column = 'math_score'
            logging.info("Read the train & test data completed")

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)

            pre_proccessor = self.get_data_transformation_obj()
            logging.info("transformed the numerical and categorical feautres")

            processed_train_data = pre_proccessor.fit_transform(input_feature_train_df)
            processed_test_data = pre_proccessor.transform(input_feature_test_df)
            logging.info("transformation of train and test data feature is completed")

            train_arr = np.c_[processed_train_data,np.array(train_df[target_column])]
            test_arr = np.c_[processed_test_data, np.array(test_df[target_column])]

            save_obj(self.data_transformation_config.preprocessing_obj_path, pre_proccessor)
            logging.info("saved the pickle file")

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessing_obj_path
            )
        
        except Exception as e:
            raise FraudException(e,sys)