# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:55:31 2022

@author: Yeshiwas
"""

import json

import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import matplotlib.pyplot as plt

font_name = "Nyala"
DATA_PATH = "datasets/MFCCs_dataset.json"
SAVED_MODEL_PATH = "models/CNN_MFCCs_new.h5"

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


def get_data_splits(data_path, test_size=0.2, test_validation=0.1):
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


def build_model(input_shape, learning_rate, loss="sparse_categorical_crossentropy"):
    # build  network
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu",
                                  padding='same', input_shape=input_shape,
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=keras.regularizers.L2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1),
                                  kernel_initializer='glorot_uniform',
                                  padding='same', kernel_regularizer=keras.regularizers.L2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(0.2))

    # conv layer 3
    model.add(keras.layers.Conv2D(128, (3, 3), activation="relu",
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=keras.regularizers.L2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(0.2))

    # flatten the output feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORD, activation="softmax"))

    # compile the model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=['acc'])

    # print the model overview
    model.summary()

    return model


def plot_acc_history(history):
    fig, axs = plt.subplots(2, figsize=(10, 8))
    # create accuracy subplot
    axs[0].plot(history.history['acc'], label="Train accuracy")
    axs[0].plot(history.history['val_acc'], label="Test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    plt.show()


def plot_err_history(history):
    fig, axs = plt.subplots(2, figsize=(10, 8))
    # create loss subplot
    axs[1].plot(history.history["loss"], label="Train error")
    axs[1].plot(history.history["val_loss"], label="Test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def main():

    # load train/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # (#segments, #cofficients =13, #channels(1 means mono)
    model = build_model(input_shape, LEARNING_RATE)

    # Train the model
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    print(f"Test Error: {test_error}, Test Accuracy: {test_accuracy}")
    
    
   # Creating confusion matrix
    cm = confusion_matrix(predictions, y_test)
    plt.figure(figsize=(10, 10))
    ### Set font name
    sns.set(font=font_name)
    sns.heatmap(cm, annot=True, xticklabels=trans, yticklabels=trans, fmt='d',
                cmap=plt.cm.Blues, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    target_name = ["ሁሉንም ምረጥ",
        "ለጥፍ",
        "ሙዚቃ ክፈት",
        "ሙዚቃ ዝጋ",
        "ምረጥ",
        "ቀልብስ",
        "ቀዳሚ",
        "ቀጣይ",
        "ቁረጥ",
        "ቅንብሩን ክፈት",
        "ቅዳ",
        "ቆልፍ",
        "በመጠን ደርድር",
        "በስም ደርድር",
        "በቀን ደርድር",
        "በትልቅ አዶ አሳይ",
        "ኖትፓድ ክፈት",
        "ኖትፓድ ዝጋ",
        "አሰይፍ",
        "አሳንስ",
        "አሳድግ",
        "አስምርበት",
        "አስቀምጥ",
        "አትም",
        "አዲስ ክፈት",
        "አድምቅ",
        "አድስ",
        "ካሜራ ክፈት",
        "ካሜራ ዝጋ",
        "ወደቀኝ ተጓዝ",
        "ዝጋ",
        "ዩቱብ ክፈት",
        "ደግመህ ስራ",
        "ድምቀት ቀንስ",
        "ድምቀት ጨምር",
        "ድምጽ ቀንስ",
        "ድምጽ ጨምር",
        "ጎግል ክፈት",
        "ፎቶ ክፈት",
        "ፎቶ ዝጋ"]
    print(classification_report(predictions, y_test, target_names=target_name))
    plt.show()



    # save the model
    model.save(SAVED_MODEL_PATH)

    # plot history
    plot_acc_history(history)
    plot_err_history(history)


if __name__ == "__main__":
    main()







