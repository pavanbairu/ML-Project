import os
import sys
import numpy as np
import pandas as pd

from src.utils import load_object
from src.exception import StudentException
from src.logger import logging


class Predict:

    def __init__(self):
        pass

    def predict_data(self, feature):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessing.pkl")

            logging.info("get the paths of models")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            logging.info("load the models")
            scaled_data = preprocessor.transform(feature)
            result = model.predict(scaled_data)

            return result
        
        except Exception as e:
            raise StudentException(e,sys)

class StudentData:
    def __init__(self,
                 gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        data = {
            "gender" : [self.gender],
            "race_ethnicity" : [self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch" : [self.lunch],
            "test_preparation_course" : [self.test_preparation_course],
            "reading_score" : [self.reading_score],
            "writing_score" : [self.writing_score]           
        }
        return  pd.DataFrame(data=data)    