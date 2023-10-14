# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:20:36 2022

@author: Yeshiwas
"""

import keras
from keras import utils
from keras.callbacks import History
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import json

font_name = "Nyala"
DATA_PATH = "MFCCs_dataset.json"
SAVED_MODEL_PATH = "CNN_MFCCs.h5"

LEARNING_RATE = 0.0001
EPOCHS = 100 # it tells how many time the network will see the whole dataset
BATCH_SIZE = 32
NUM_KEYWORD = 40  # total number of basic keywords/ command keywords 


def load_dataset(data_path):
    with open(data_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)
    # extract inputs and targets(outputs)
    global trans

    X = np.array(data["MFCCs"])
    y = np.array(data["Labels"])
    trans = np.array(data['Transcription'])


    return X, y


def get_data_splits(data_path, test_size=0.2, test_validation=0.2):
    # load the dataset
    X, y = load_dataset(data_path)
    print("Data", X.shape)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) # train/test splits
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                   test_size=test_validation, random_state=42) # train/validation splits
    """ 10% of dataset is used for testing purpose
    10 % of the remaining(90%) is used for validation : which means 9 % dataset
    """
    print("Training Data")
    print(X_train.shape)
    print("Validation Data")
    print(X_validation.shape)
    print("Testing Data ")
    print(X_test.shape)
    # convert the input from 2D to 3D arrays # (#segments, #mfccs(13), 1)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, loss="sparse_categorical_crossentropy", init_mode='uniform'):
    # build  network
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu",
                                  padding='same', input_shape=input_shape,
                                  kernel_initializer=init_mode,
                                  kernel_regularizer=keras.regularizers.L2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1),
                                  kernel_initializer=init_mode,
                                  padding='same', kernel_regularizer=keras.regularizers.L2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(0.2))

    # conv layer 3
    model.add(keras.layers.Conv2D(128, (3, 3), activation="relu",
                                  kernel_initializer=init_mode,
                                  kernel_regularizer=keras.regularizers.L2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(0.2))

    # flatten the output feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.4))  # to tackle over_fitting 0.3 is a dropout probability

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORD, activation="softmax"))

    # compile the model
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])

    # print the model overview
    model.summary()

    return model






def main():
    # load train/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # (#segments, #cofficients =13, #channels(1 means mono)
    model = build_model(input_shape, LEARNING_RATE)
    
    ### Hyper Parameter Tuning using GridSearch 
    seed = 7
    np.random.seed(seed)
    model_cv = KerasClassifier(build_fn=build_model, input_shape=input_shape,
                               verbose=1)
    init_mode = ['normal', 'uniform', 'glorot_uniform']
    batches=[32,64]
    epochs = [80,100]
    param_grid = dict(batch_size=batches, epochs=epochs, init_mode=init_mode)
    grid = GridSearchCV(estimator=model_cv, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    
    ###Print grid_results
    print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4}, using {param}')
    
    
if __name__ == "__main__":
    main()






