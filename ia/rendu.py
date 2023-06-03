# created by
# FAYS Matthieu
# FONTAINE Thomas

import csv

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense
from keras.layers import LSTM
from keras.models import Model
from keras.utils import to_categorical
from tensorflow import keras
from tqdm import tqdm

#path to data
file_path_train = 'ECG200_TRAIN.tsv'
file_path_test = 'ECG200_TEST.tsv'


def cnn():
    # created by
    # FONTAINE Thomas

    # -------------------------------- définition des fonctions --------------------------------

    def load_data():
        global file_path_train
        global file_path_test

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        with open(file_path_train) as file_train:
            tsv_file = csv.reader(file_train, delimiter="\t")
            for line in tsv_file:
                x_train.append([float(i) for i in line[1::]])
                y_train.append(float(line[0]))

        with open(file_path_test) as file_test:
            tsv_file = csv.reader(file_test, delimiter="\t")
            for line in tsv_file:
                x_test.append([float(i) for i in line[1::]])
                y_test.append(float(line[0]))

        # Work on the data in order to use it easily (normalization + reshape)
        x_train = (np.array(x_train) + 2) / 6
        x_test = (np.array(x_test) + 2) / 6
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        y_train = np.where(y_train == -1, 0, 1)
        y_test = np.where(y_test == -1, 0, 1)

        x_train = x_train.reshape(-1, 96, 1, 1)
        x_test = x_test.reshape(-1, 96, 1, 1)
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        return x_train, y_train, x_test, y_test

    def create_model(x_train):
        padding = 'same'
        input_shape = x_train.shape[1:]
        stride = 1
        kernel_size = (3, 3)  # (3,3)
        filters = 32  # 8
        activation = 'relu'
        pool_size = (2, 2)
        nb_classes = 2
        input_layer = keras.layers.Input(input_shape)

        hidden_conv_layer_1 = keras.layers.Conv2D(filters=filters,
                                                  kernel_size=kernel_size, strides=stride,
                                                  padding=padding, activation=activation)(input_layer)

        pooling_conv_layer_1 = keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride,
                                                         padding=padding)(hidden_conv_layer_1)

        dropout_1 = keras.layers.Dropout(0.25)(pooling_conv_layer_1)

        hidden_conv_layer_2 = keras.layers.Conv2D(filters=filters,
                                                  kernel_size=kernel_size, strides=stride,
                                                  padding=padding, activation=activation)(dropout_1)

        pooling_conv_layer_1 = keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride,
                                                         padding=padding)(hidden_conv_layer_2)

        dropout_2 = keras.layers.Dropout(0.25)(pooling_conv_layer_1)

        flattened_layer_1 = keras.layers.Flatten()(dropout_2)

        dense_layer_1 = keras.layers.Dense(128, activation=activation)(flattened_layer_1)

        dropout_3 = keras.layers.Dropout(0.5)(dense_layer_1)

        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(dropout_3)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        learning_rate = 0.1
        optimizer_algo = keras.optimizers.SGD(learning_rate=learning_rate)

        model.summary()

        cost_function = keras.losses.categorical_crossentropy
        model.compile(loss=cost_function, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        return model

    def learn(model, model_name, x_train, y_train):
        mini_batch_size = 10
        nb_epochs = 150

        percentage_of_train_as_validation = 0.2
        history = model.fit(x_train, y_train, batch_size=mini_batch_size,
                            epochs=nb_epochs, verbose=True,
                            validation_split=percentage_of_train_as_validation,
                            callbacks=[keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)])

        history_dict = history.history
        loss_train_epochs = history_dict['loss']
        loss_val_epochs = history_dict['val_loss']

        plt.figure()
        plt.plot(loss_train_epochs, color='blue', label='train_loss')
        plt.plot(loss_val_epochs, color='red', label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('epoch-loss.pdf')
        plt.show()

    def test(model_name, x_train, y_train, x_test, y_test):
        model = keras.models.load_model(model_name)

        loss, acc = model.evaluate(x_train, y_train, verbose=False)

        print("La précision du modèle sur l'ensemble d'entrainements est de:", acc)

        print("La loss du modèle sur l'ensemble d'entrainements est de:", loss)

        loss, acc = model.evaluate(x_test, y_test, verbose=False)

        print("La précision du modèle sur l'ensemble de tests est de:", acc)

        print("La loss du modèle sur l'ensemble de tests est de:", loss)

    # -------------------------------- run --------------------------------

    x_train, y_train, x_test, y_test = load_data()

    model = create_model(x_train)

    learn(model, 'best-model.h5', x_train, y_train)

    test('best-model.h5', x_train, y_train, x_test, y_test)


def rnn():
    # created by
    # FAYS Matthieu

    # -------------------------------- définition des fonctions --------------------------------

    def load_data():
        global file_path_train
        global file_path_test

        X_train = []
        X_test = []

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        with open(file_path_train) as file_train:
            tsv_file = csv.reader(file_train, delimiter="\t")
            for line in tsv_file:
                x_train.append([float(i) for i in line[1::]])
                y_train.append((float(line[0]) + 1) / 2)

        with open(file_path_test) as file_test:
            tsv_file = csv.reader(file_test, delimiter="\t")
            for line in tsv_file:
                x_test.append([float(i) for i in line[1::]])
                y_test.append((float(line[0]) + 1) / 2)

        for i in x_train:
            amplitude = max(i) - min(i)
            X_train.append([k / amplitude for k in i])

        for i in x_test:
            amplitude = max(i) - min(i)
            X_test.append([k / amplitude for k in i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test

    def create_model(k, X_train, y_train):

        # Define the input shape
        input_shape = (X_train.shape[1:])  # Assuming X_train has shape (num_samples, timesteps)

        # Create the input layer
        layers = [Input(shape=input_shape)]

        for i in k[:-1:]:
            layers.append(LSTM(units=i, return_sequences=True, stateful=False)(layers[-1]))

        layers.append(LSTM(units=k[-1])(layers[-1]))

        # Create the output layer
        output_layer = Dense(units=y_train.shape[1], activation='sigmoid')(layers[-1])

        # Create the model
        model = Model(inputs=layers[0], outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def learn(model, n, X_train, y_train, X_test, y_test):

        # Train the model
        for _ in tqdm(range(n), desc="Progression"):
            model.fit(X_train, y_train, batch_size=10, epochs=1, validation_data=(X_test, y_test),
                      verbose=0)
            model.reset_states()

    def test(model, X_train, y_train, X_test, y_test):

        # Evaluate the model
        loss, accuracy = model.evaluate(X_train, y_train, verbose=0)

        print("La précision du modèle sur l'ensemble d'entrainements est de:", accuracy)

        print("La loss du modèle sur l'ensemble d'entrainements est de:", loss)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        print("La précision du modèle sur l'ensemble de tests est de:", accuracy)

        print("La loss du modèle sur l'ensemble de tests est de:", loss)

    # -------------------------------- run --------------------------------

    X_train, y_train, X_test, y_test = load_data()

    model = create_model([32, 32], X_train, y_train)

    learn(model, 200, X_train, y_train, X_test, y_test)

    test(model, X_train, y_train, X_test, y_test)


print("-------------------------------- cnn --------------------------------")
cnn()

print("-------------------------------- rnn --------------------------------")
rnn()
