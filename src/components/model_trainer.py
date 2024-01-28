import os
import sys
import numpy as np 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging

# Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Ensemble models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

# Regularization models
from sklearn.linear_model import Ridge, Lasso

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

from src.utils import evaluate_models, save_object
# Model Evaluation 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("spliting training training and test input data")
            x_train, y_train, x_test, y_test= [
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            ]

            models= {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor" :KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict= evaluate_models(x_train= x_train, y_train= y_train, x_test= x_test,
                                                y_test= y_test, models= models)
            highest_r_squared= 0
            best_model:str= ''
            for model_name, report in model_report.items():
                if report['r2_score_test']>highest_r_squared:
                    highest_r_squared= report['r2_score_test']
                    best_model= model_name

            if highest_r_squared< 0.6:
                raise CustomException("No best model found")
                logging.info("No best model found")
            
            bestModel= models[best_model].fit(x_train, y_train)

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= bestModel
            )

            predicted= bestModel.predict(x_test)

            return r2_score(y_test, predicted)
        
        except Exception as e:
            raise CustomException(e, sys)




