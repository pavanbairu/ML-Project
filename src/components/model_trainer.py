import os
import sys

from dataclasses import dataclass
from src.exception import StudentException
from src.logger import logging
from src.utils import save_obj, evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pickle

@dataclass
class ModelTrainerConfig:
    model_trainer_path=os.path.join("artifacts", "model.pkl")


class ModelTrainer():

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def load_object(self, file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            raise StudentException(e, sys)
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("split the train and test array completed")

            models = {
                "LinearRegression" : LinearRegression(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose=False)
            }

            params={
                "Linear Regression":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                }, 
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },        
            }

            model_report, trained_models = evaluate_models(X_train, X_test, y_train, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = [model for model, score in model_report.items() if score == best_model_score]
            best_model_name = best_model_name[0]

            best_model = trained_models[best_model_name]

            save_obj(self.model_trainer_config.model_trainer_path, best_model)

            new_model = self.load_object(self.model_trainer_config.model_trainer_path)

            return f"best_model : {best_model_name} and score : {best_model_score}"

        except Exception as e:
            raise StudentException(e, sys)