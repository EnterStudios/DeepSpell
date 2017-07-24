# encoding: utf-8
'''

Use hyperas
to tune the hyperparameters of Keras deep learning models, such as: learning rate, dropout rate, epochs.

Based in part on:
http://maxpumperla.github.io/hyperas/

'''

from __future__ import print_function, division, unicode_literals

import os
from hashlib import sha256
import json
import datetime
from numpy.random import seed as random_seed
from keras_spell import CharacterTable, read_top_chars, generate_question, _vectorize, generate_model, Configuration, print_random_predictions

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe, space_eval

from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout, recurrent
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

random_seed(123) # Reproducibility

CONFIG = Configuration()
DIGEST = sha256(json.dumps(CONFIG.__dict__, sort_keys=True)).hexdigest()

# Parameters for the dataset
MIN_INPUT_LEN = 5
AMOUNT_OF_NOISE = 0.2 / CONFIG.max_input_len
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
PADDING = "☕"


def data():
    DATA_FILES_PATH = "~/DeepSpell/Downloads/data"
    DATA_FILES_FULL_PATH = os.path.expanduser(DATA_FILES_PATH)
    NEWS_FILE_NAME_SPLIT = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split")
    NEWS_FILE_NAME_SAMPLE = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.sample_{}")
    CHAR_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "char_frequency.json")
    SAVED_MODEL_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "keras_spell_hyperas_e{}.h5") # an HDF5 file


    size = 0.0001
    answers = open(NEWS_FILE_NAME_SAMPLE.format(size)).read().decode('utf-8').split("\n")
    qa_tuples = [generate_question(answer) for answer in answers]
    questions = [qa[0][::-1] for qa in qa_tuples]

    chars = read_top_chars()
    ctable = CharacterTable(chars)
    X, y = _vectorize(questions, answers, ctable)

    # Explicitly split train and test data
    all_data = X.shape[0]
    train_size = 0.7
    split_at = int(round(all_data * train_size))

    x_train = X[:split_at]
    x_test = X[split_at:]

    y_train = y[:split_at]
    y_test = y[split_at:]

    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    CONFIG = Configuration()
    chars = read_top_chars()
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of hidden_size
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(CONFIG.input_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, input_shape=(None, len(chars)), kernel_initializer=CONFIG.initialization,
                                 return_sequences=layer_number + 1 < CONFIG.input_layers))
        model.add(Dropout(CONFIG.amount_of_dropout))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(CONFIG.max_input_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(CONFIG.output_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, return_sequences=True, kernel_initializer=CONFIG.initialization))
        model.add(Dropout(CONFIG.amount_of_dropout))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), kernel_initializer=CONFIG.initialization)))
    model.add(Activation('softmax'))

    model.compile(loss=CONFIG.loss, optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size={{choice([100, 200, 300, 400])}},
              epochs={{choice([100, 200, 250, 300])}},
              verbose=2,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    print('Random Predictions:')
    ctable = CharacterTable(read_top_chars())
    print_random_predictions(model, ctable, x_test, y_test)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime("%Y-%m-%d %H:%M"))
    print("Starting Hyperparameter Optimization.")
    
    DATA_FILES_PATH = "~/DeepSpell/Downloads/data"
    DATA_FILES_FULL_PATH = os.path.expanduser(DATA_FILES_PATH)
    SAVED_MODEL_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "keras_spell_hyperas_e{}.h5") # an HDF5
    trials = Trials()
    best_run, best_model, space = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=trials,
                                          eval_space=True,
                                          return_space=True)
    X_train, Y_train, X_test, Y_test = data()
   
    print("Finished Hyperparameter Optimization.")
    end = datetime.datetime.now()
    delta = end - start
    print("Hyperparameter Optimization took {:.1f} seconds.".format(delta.total_seconds()))

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Saving best performing model...")
    best_model.save(SAVED_MODEL_FILE_NAME)
