import os
from mlflow import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlflow.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    ## note: you can add different transformation techniques such a scaler, PCA and all
    ## you can perform all kinds of EDA in ml cycle here before passing this data to the model
    # i am only adding train_test_spliting cz this data is already cleaned up

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)