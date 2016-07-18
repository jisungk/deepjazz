'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Builds an LSTM, a type of recurrent neural network (RNN). 

Code was built while significantly referencing public examples from the
Keras documentation on GitHub:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

''' Build a 2-layer LSTM from a training corpus '''
def build_model(corpus, val_indices, max_len, N_epochs=128):
    # number of different values or words in corpus
    N_values = len(set(corpus))

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print('nb sequences:', len(sentences))

    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1

    # build a 2 stacked LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, N_values)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(N_values))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X, y, batch_size=128, nb_epoch=N_epochs)

    return model