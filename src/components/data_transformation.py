import pandas as pd
import sys
import os
from dataclasses import dataclass
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src import utils


@dataclass
class DataTranformationConfig:
    data_path = "/home/sai/Documents/end_to_end_ML/mlproj/data/"
    preprocessor_obj_file = os.path.join(data_path, "preprocessor.pkl")

class DataTransformer(object):
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("creating Numerical and Categorical Transformation object")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            logging.info("Numerical and Categorical Transformation object ready")

            return preprocessor

        except Exception as e:
            logging.error("Cannot do preprocessing")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("reading train and test CSV files")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("getting data transformation object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column = "math score"


            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("tranforming train and test df")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            utils.save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file)

        except Exception as e:
            logging.error("Cannot do data transformation")
            raise CustomException(e, sys)