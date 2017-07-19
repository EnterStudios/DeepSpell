# encoding: utf-8
'''

Use the grid search capability from the scikit-learn python machine learning library
to tune the hyperparameters of Keras deep learning models, such as: learning rate, dropout rate, epochs.

Based in part on:
http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

'''

from __future__ import print_function, division, unicode_literals

import os
from collections import Counter
from hashlib import sha256
import json
import numpy as np
from numpy.random import seed as random_seed, shuffle as random_shuffle
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from keras_spell_local import CharacterTable, read_top_chars, generate_question, _vectorize

from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout, recurrent
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier


random_seed(123) # Reproducibility


class Configuration(object):
    """Dump stuff here"""

CONFIG = Configuration()

# Parameters for the model:
CONFIG.input_layers = 2
CONFIG.output_layers = 2
CONFIG.amount_of_dropout = 0.2
CONFIG.hidden_size = 500
CONFIG.initialization = "he_normal" # : Gaussian initialization scaled by fan-in (He et al., 2014)
CONFIG.loss = 'binary_crossentropy'
CONFIG.optimizer = 'adam'

# Parameters for text:
CONFIG.number_of_chars = 100
CONFIG.max_len = 60
CONFIG.inverted = True

# Parameters for the training:
CONFIG.batch_size = 240 # As the model changes in size, play with the batch size to best fit the process in memory
CONFIG.epochs = 500 # due to mini-epochs.
CONFIG.steps_per_epoch = 934 # samples: only 2-gram  # This is a mini-epoch. Using News 2013 an epoch would need to be ~60K.
CONFIG.validation_steps = 10
CONFIG.number_of_iterations = 10

DIGEST = sha256(json.dumps(CONFIG.__dict__, sort_keys=True)).hexdigest()

# Parameters for the dataset
MIN_INPUT_LEN = 5
AMOUNT_OF_NOISE = 0.2 / CONFIG.max_len
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
PADDING = "☕"

DATA_FILES_PATH = "~/DeepSpell/Downloads/data"
DATA_FILES_FULL_PATH = os.path.expanduser(DATA_FILES_PATH)
NEWS_FILE_NAME_SPLIT = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split")
NEWS_FILE_NAME_SAMPLE = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.sample_{}")
CHAR_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "char_frequency.json")
SAVED_MODEL_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "keras_spell_grid_search_e{}.h5") # an HDF5 file


def get_sample_data(sample_size):
    """Set asside sample data for grid search.
    sample_size: float
    """
    all_data = open(NEWS_FILE_NAME_SPLIT).read().decode('utf-8').split("\n")
    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Shuffling data...')
    random_shuffle(all_data)

    # Explicitly set apart sample_size as sample
    split_at = int(round(len(all_data) * sample_size))
    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Saving sample...')
    with open(NEWS_FILE_NAME_SAMPLE.format(sample_size), "wb") as output_file:
        output_file.write("\n".join(all_data[:split_at]).encode('utf-8'))

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Done.')


def vectorize_text(filename, chars=None):
    """Vectorize the text into questions (X) and expected answers (y)"""

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Reading answers and generating questions...')
    answers = open(filename).read().decode('utf-8').split("\n")
    qa_tuples = [generate_question(answer) for answer in answers]
    questions = [qa[0] for qa in qa_tuples]

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Vectorization...')
    chars = chars or CHARS
    ctable = CharacterTable(chars)
    X, y = _vectorize(questions, answers, ctable)

    print(X.shape)
    print(y.shape)
    return X, y


def generate_model(output_len, chars=None):
    """Generate the model"""
    print('Build model...')
    chars = chars or CHARS
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of hidden_size
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(CONFIG.input_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, input_shape=(None, len(chars)),
                                 kernel_initializer=CONFIG.initialization,
                                 return_sequences=layer_number + 1 < CONFIG.input_layers))
        model.add(Dropout(CONFIG.amount_of_dropout))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(output_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(CONFIG.output_layers):
        model.add(recurrent.LSTM(CONFIG.hidden_size, return_sequences=True, kernel_initializer=CONFIG.initialization))
        model.add(Dropout(CONFIG.amount_of_dropout))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), kernel_initializer=CONFIG.initialization)))
    model.add(Activation('softmax'))

    model.compile(loss=CONFIG.loss, optimizer=CONFIG.optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # get dataset
    size = 0.00001
    #get_sample_data(size)

    # split data into X and Y
    X, Y = vectorize_text(NEWS_FILE_NAME_SAMPLE.format(size), chars=read_top_chars())

    # create model
    model = KerasRegressor(build_fn=generate_model, output_len=CONFIG.max_len, chars=read_top_chars())

    # define the grid search parameters
    # those should be added as new parameters to the generate_model(), or existent on the Keras fit() already.
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    print('Grid Search for model...')
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


