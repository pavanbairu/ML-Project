import os
import sys
import pickle
from typing import Tuple, Dict

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import StudentException
from src.logger import logging


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise StudentException(e, sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models, params)-> Tuple[Dict, Dict]:
    try:
        reports = {}
        trained_models = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param_grid = list(params.values())[i]

            
            gs = GridSearchCV(model, param_grid=param_grid, cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_trian_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            score = r2_score(y_test, y_test_pred)

            reports[list(models.keys())[i]] = score
            trained_models[list(models.keys())[i]] = model

        logging.info("Hyper parameter tuning completed")
        logging.info("trained all the models with best params and obtained the scores")

        return (reports, trained_models)
    except Exception as e:
        raise StudentException(e, sys)
    
def load_object(path):
    try:
        with open(path, 'rb') as input:
            model = pickle.load(input)
        
        return model
    except Exception as e:
        raise StudentException(e, sys)