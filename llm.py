import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.saving import load_model,save_model


def generate_model(X_tr, X_val, y_tr, y_val, vocab):
    model = Sequential([
        Embedding(vocab, 100, input_length=30, trainable=True),
        Dropout(0.2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(vocab, activation='softmax')
    ])

    print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    # fit the model
    model.fit(X_tr, y_tr, epochs=60, verbose=2, validation_data=(X_val, y_val))

    model.save("llm.keras")

    return model

def model_train_on_new(X_tr, X_val, y_tr, y_val, vocab):
    model = load_model("llm.keras")

    model.fit(X_tr, y_tr, epochs=60, verbose=2, validation_data=(X_val, y_val))

    model.save("llm.keras")