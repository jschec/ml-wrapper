from sklearn.model_selection import train_test_split
from typing import Any
import pandas as pd

class Dataset():
   #TODO
    """
    The Dataset class provides a means to aggregate numeric and categorical
    features and processes them into training and test sets  
    """

    # Non-target training set
    __x_train = None
    # Non-target test set
    __x_test = None
    # Target training set
    __y_train = None
    # Target test set
    __y_test = None
    # Numeric features for this Dataset
    __num_features = []
    # Categorical features for this Dataset
    __cat_features = []
    # Name of target feature
    __target_feature = ""

    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        if kwargs.keys() >= {"config", "data"}:
            config = kwargs.get("config")
            self.__cat_features = config.get("categorical_feature")
            self.__num_features = config.get("numerical_feature")
            self.__target_feature = config.get("target_feature")
            data = kwargs.get("data")

            if "test_size" in config:
                tst_size = config.get("test_size")
                self.__create_training_set(data, tst_size)
            else:
                self.__create_training_set(data, 0.2)
        else:
            raise ValueError

    @property
    def x_train(self) -> Any:
        return self.__x_train

    @property
    def x_test(self) -> Any:
        return self.__x_test

    @property
    def y_train(self) -> Any:
        return self.__y_train

    @property
    def y_test(self) -> Any:
        return self.__y_test

    @property
    def num_features(self) -> list:
        return self.__num_features

    @num_features.setter
    def num_features(self, num_features: list) -> None:
        self.__num_features = num_features

    @property
    def cat_features(self) -> list:
        return self.__cat_features

    @cat_features.setter
    def cat_features(self, cat_features: list) -> None:
        self.__cat_features = cat_features

    @property
    def target_feature(self) -> str:
        return self.target_feature

    @target_feature.setter
    def target_feature(self, target_feature: str) -> None:
        self.__target_feature = target_feature

    def __create_training_set(self, data: Any, test_size: float) -> None:
        data[self.__num_features] = data[self.__num_features].apply(pd.to_numeric, errors='coerce')
        target = data[self.__target_feature]

        #does this actually work?
        nontarget_feature_columns = list(self.__num_features) + list(self.__cat_features)
        nontarget_feature_columns.remove(self.__target_feature)

        target = target.astype('int64')
        features = data[nontarget_feature_columns]
        #TODO make test size in config
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(features, target, test_size=test_size)
