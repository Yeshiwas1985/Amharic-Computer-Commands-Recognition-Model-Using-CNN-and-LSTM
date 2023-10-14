# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:20:36 2022

@author: Yeshiwas
"""

import keras
from keras.callbacks import History
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
import json

font_name = "Nyala"
DATA_PATH = "datasets/MFCCs_dataset.json"
SAVED_MODEL_PATH = "LSTM_MFCCs.h5"

LEARNING_RATE = 0.0001
EPOCHS = 100  # it tells how many times the network will see the whole dataset
BATCH_SIZE = 32
NUM_KEYWORD = 40  # total number of basic keywords/command keywords

def load_dataset(data_path):
    with open(data_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["Labels"])
    trans = np.array(data['Transcription'])

    return X, y

def get_data_splits(data_path, test_size=0.2, test_validation=0.2):
    # load the dataset
    X, y = load_dataset(data_path)
    print("Data", X.shape)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)  # train/test splits
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation,
                                                                    random_state=42)  # train/validation splits
    """ 10% of the dataset is used for testing purposes
    10 % of the remaining (90%) is used for validation: which means 9 % dataset
    """
    print("Training Data")
    print(X_train.shape)
    print("Validation Data")
    print(X_validation.shape)
    print("Testing Data ")
    print(X_test.shape)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate):
    # create model
    model = keras.Sequential()

    # two LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    """
    64: number of units
    two layers in LSTM: sequence-to-sequence and sequence-to-vector
    """
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.4))

    # output layer
    model.add(keras.layers.Dense(19, activation="softmax"))
    """19 is the no of different commands to become neurons,
     softmax activation is used for multiclass classification
    """

    # print the model overview
    model.summary()

    return model

def main():
    # load train/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])  # RNN expects 2D array (134, 13)
    model = build_model(input_shape, LEARNING_RATE)

    # Hyperparameter tuning using GridSearch
    seed = 7
    np.random.seed(seed)
    model_cv = KerasClassifier(build_fn=build_model, input_shape=input_shape, learning_rate=LEARNING_RATE, verbose=1)
    batches = [32, 64]
    epochs = [80, 100]
    param_grid = dict(batch_size=batches, epochs=epochs)
    grid = GridSearchCV(estimator=model_cv, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Print grid_results
    print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4}, using {param}')

if __name__ == "__main__":
    main()
