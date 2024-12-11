import sys
import pandas as pd

from flask import Flask, request, jsonify

from src.exception import StudentException
from src.logger import logging
from src.pipeline.predict_pipeline import StudentData, Predict

app = Flask(__name__)

application = app

@app.route("/predict/student/data", methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return "Pavan"
        else:
            data = request.get_json()
            student_data = StudentData(
                gender=data.get('gender'),
                race_ethnicity=data.get('ethnicity'),
                parental_level_of_education=data.get('parental_level_of_education'),
                lunch=data.get('lunch'),
                test_preparation_course=data.get('test_preparation_course'),
                reading_score=float(data.get('reading_score')),
                writing_score=float(data.get('writing_score'))
            )
            df = student_data.get_data_as_dataframe()
            
            print(df)
            predict = Predict()
            result = predict.predict_data(df)
            return jsonify(f"the predicted math score is : {(result)}")
        
    except Exception as e:
        raise StudentException(e,sys)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
