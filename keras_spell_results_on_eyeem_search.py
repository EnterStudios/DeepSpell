# encoding: utf-8

from __future__ import print_function, division, unicode_literals
import yaml
import keras_spell
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model


class Configuration(object):
    def __init__(self):
        with open(".settings/config.yml", "r") as ymlfile:
            settings = yaml.load(ymlfile)

            self.input_layers = settings['model']['input_layers']
            self.output_layers = settings['model']['output_layers']
            self.amount_of_dropout = settings['model']['amount_of_dropout']
            self.hidden_size = settings['model']['hidden_size']
            self.initialization = settings['model']['initialization']
            self.number_of_chars = settings['model']['number_of_chars']
            self.max_input_len = settings['model']['max_input_len']
            self.inverted = settings['model']['inverted']

            self.batch_size = settings['training']['batch_size']
            self.epochs = settings['training']['epochs']
            self.steps_per_epoch = settings['training']['steps_per_epoch']
            self.validation_steps = settings['training']['validation_steps']
            self.number_of_iterations = settings['training']['number_of_iterations']


def vectorize(filename, chars=None):
    """Vectorize the data as numpy arrays"""
    chars = chars or keras_spell.CHARS
    ctable = keras_spell.CharacterTable(chars)

    data = open(filename).read().decode('utf-8').split("\n")
    len_of_data = len(data)
    X = np.zeros((len_of_data, CONFIG.max_input_len, ctable.size), dtype=np.bool)

    for i in xrange(len(data)):
        sentence = data.pop()
        for j, c in enumerate(sentence):
            try:
                X[i, j, ctable.char_indices[c]] = 1
            except KeyError:
                pass# Padding
    return X


def read_model_to_json(h5_filename, json_filename):
    # load h5 file
    model = load_model(h5_filename)

    # print summary
    print(model.summary())

    # serialize model to JSON
    model_json = model.to_json()
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)


def preprocess_data(list_strings, chars):
    """Pre-process the data - cleanup"""
    clean_list = []
    for line in list_strings:
        decoded_line = line.decode('utf-8')
        cleaned_line = keras_spell.clean_text(decoded_line)
        if cleaned_line and not bool(set(cleaned_line) - set(chars)):
            clean_list.append(cleaned_line)
        else:
            clean_list.append('not valid query'.decode('utf-8'))

    return clean_list


def apply_model(model_file, input_filename, output_filename):

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Reading model.')
    model = load_model(model_file)

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Reading and pre-processing input data.')
    df_input_all = pd.read_csv(input_filename)
    df_input = df_input_all
    most_popular_chars = keras_spell.read_top_chars()
    ctable = keras_spell.CharacterTable(most_popular_chars)

    # question, or original query
    questions = list(df_input.search_term_norm)
    questions_filtered = preprocess_data(questions, chars=most_popular_chars)

    questions_trimmed = []
    for line in questions_filtered:
        if len(line) > CONFIG.max_input_len:
            questions_trimmed.append(line[:CONFIG.max_input_len])
        else:
            questions_trimmed.append(line)

    X = [ctable.encode(row, maxlen=CONFIG.max_input_len) for row in questions_trimmed]

    # answer, or google correction data
    answers = list(df_input.google_correction)
    answers = preprocess_data(answers, chars=most_popular_chars)

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Scoring input data.')
    predictions = model.predict(np.array(X), verbose=1)
    guesses = [ctable.decode(row) for row in predictions]

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Saving results.')
    result_df = pd.DataFrame(np.column_stack([questions_trimmed, answers, guesses]),
                             columns=['questions', 'answers', 'guesses'])
    result_df['flag_mistake'] = np.where(result_df.questions != result_df.answers, 1, 0)
    result_df['flag_correct_guess'] = np.where(result_df.guesses == result_df.answers, 1, 0)
    result_df.to_csv(output_filename, index=0, encoding='utf-8') # '/root/DeepSpell/Downloads/results.csv'

    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + 'Printing summary and sample output:')
    print(pd.crosstab(result_df.flag_mistake, result_df.flag_correct_guess))
    print(result_df.head(10))


if __name__ == '__main__':
    print(datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S") + " " + "Starting.")
    CONFIG = Configuration()

    apply_model(model_file="/Users/dania/DeepSpell/models/keras_spell_e699.h5",
                input_filename="/Users/dania/captainCrunch/search/spell_correct/df_google_spell_check_results.csv",
                output_filename="/Users/dania/captainCrunch/search/spell_correct/model_e699_results.csv")