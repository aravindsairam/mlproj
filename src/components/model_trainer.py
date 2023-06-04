import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression, Ridge,Lasso

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

from src import utils


@dataclass
class ModelTrainerConfig:
    data_path = "/home/sai/Documents/end_to_end_ML/mlproj/data/"
    trained_model_file = os.path.join(data_path, "model.pkl")

    models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoost Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
    }

    params={
            "LinearRegression":{},
            "Lasso":{
                'alpha': [0.1, 0.5, 1.0]
            },
            "Ridge":{
                'alpha': [0.1, 0.5, 1.0]
            },
            "K-Neighbors Regressor":{
                'n_neighbors': [2, 5, 10, 15],
            },
            "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest Regressor":{
                'n_estimators': [8,16,32,64,128,256]
            },
            "XGBRegressor":{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "CatBoost Regressor":{
                'depth': [6,8,10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor":{
                'learning_rate':[.1,.01,0.5,.001],
                'n_estimators': [8,16,32,64,128,256]
            }        
        }

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_obj_path):
        try:
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            
            logging.info("Evaluating models")

            models_report = utils.evaluate_model(x_train, y_train, x_test, y_test, self.model_trainer_config.models, self.model_trainer_config.params)

            best_model_score = max(sorted(models_report.values()))

            best_model_name = list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            best_model = self.model_trainer_config.models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No model found", sys)
            
            logging.info(f"Best model found is {best_model_name} with R2 score of {best_model_score}")

            utils.save_object(self.model_trainer_config.trained_model_file, best_model)

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error("Model not trained")
            raise CustomException(e, sys)