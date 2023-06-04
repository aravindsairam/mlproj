import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Pickle file saved in {file_path}")

    except Exception as e:
        logging.error("pickle file not saved")
        raise CustomException(e, sys)
    
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)

            train_model_score_ = r2_score(y_train, y_train_predict)
            test_model_score = r2_score(y_test, y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.error("Cannot evaluate models")
        raise CustomException(e, sys)