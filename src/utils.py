import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
# Model Evaluation 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    model_performance_report= {}
    for model_name, model in models.items():
        # training
        model.fit(x_train, y_train)

        # prediction
        y_train_pred= model.predict(x_train)
        y_test_pred= model.predict(x_test)

        # Evaluate model
        mae_train, mse_train, r2_square_train= evaluate_model(y_train, y_train_pred)
        mae_test, mse_test, r2_square_test= evaluate_model(y_test, y_test_pred)
        
        model_performance_report[model_name]= {
            'mae_train': mae_train,
            'mse_train': mse_train,
            'r2_score_train': r2_square_train * 100,
            'mae_test': mae_test,
            'mse_test': mse_test,
            'r2_score_test': r2_square_test * 100
        }
    return model_performance_report

def evaluate_model(actual, predicted):
    mae= mean_absolute_error(actual, predicted)
    mse= mean_squared_error(actual, predicted)
    rmse= np.sqrt(mse)
    r2_square= r2_score(actual, predicted)
    return mae, rmse, r2_square