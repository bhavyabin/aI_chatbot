import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


def generate_model(X_tr, X_val, y_tr, y_val, vocab):
    model = Sequential([
        Embedding(vocab, 60, input_length=30, trainable=True),
        GRU(150, recurrent_dropout=0.1, dropout=0.1),
        Dense(vocab, activation='relu'),
        Dense(vocab, activation='softmax')
    ])

    print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    # fit the model
    model.fit(X_tr, y_tr, epochs=100, verbose=2, validation_data=(X_val, y_val))

    model.save("llm.h5")

    return model