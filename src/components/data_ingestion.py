# reading data from sources

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

'''
DataIngestionConfig class:

This class is defined using the @dataclass decorator, indicating that it is meant to hold data.
It has three attributes (train_data_path, test_data_path, and raw_data_path) representing file paths for training data, testing data, and raw data, respectively.
Default values for these paths are specified using os.path.join to concatenate the 'artifacts' directory with the corresponding file names.
'''

@dataclass
class DataIngestionConfig:
    train_data_path:str= os.path.join('artifacts', "train.csv")
    test_data_path:str= os.path.join('artifacts', "test.csv")
    raw_data_path:str= os.path.join('artifacts', "data.csv")


'''
DataIngestion class:

This class is responsible for handling the data ingestion process.
It has an __init__ method that initializes an instance of DataIngestionConfig.
The initiate_data_ingestion method is the main function for data ingestion.
It begins by logging an informational message and attempting to read a CSV file ('notebook\data\stud.csv') into a DataFrame (df).
It then creates the necessary directories for the output files using os.makedirs.
The raw data is saved to a CSV file specified by raw_data_path.
The dataset is split into training and testing sets using train_test_split from scikit-learn.
Both the training and testing sets are saved to CSV files specified by train_data_path and test_data_path, respectively.
Finally, the method logs the completion of the data ingestion and returns the paths of the training and testing data.

Exception Handling:

The code includes a try-except block to catch any exceptions that may occur during data ingestion.
If an exception is caught, it raises a CustomException with the original exception and the sys module.
'''
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df= pd.read_csv('notebook\data\stud.csv')
            logging.info("read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index= False, header=True)

            logging.info("initiated train test split")

            train_set, test_set= train_test_split(df, test_size=0.2, random_state=101)

            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
         


if __name__=='__main__':
    obj= DataIngestion()
    train_data, test_data= obj.initiate_data_ingestion()

    data_transformation= DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)