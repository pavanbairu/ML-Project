import os
import sys
import pickle
from typing import Tuple, Dict

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import FraudException
from src.logger import logging


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise FraudException(e, sys)
    