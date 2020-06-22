import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series)-window_size):
        X += [series[i:i+window_size]]
        y += [series[i+window_size]]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    RNN_model = Sequential()
    RNN_model.add(LSTM(5, input_shape=(window_size, 1)))
    RNN_model.add(Dense(1))
    return RNN_model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
import string
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    lowercase = list(string.ascii_lowercase)
    
    clean_text = ''.join([c if (c in lowercase or c in punctuation) else ' ' for c in text])

    return clean_text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0,len(text)-window_size,step_size):
        inputs += [text[i:i+window_size]]
        outputs += [text[i+window_size]]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    RNN_model = Sequential()
    RNN_model.add(LSTM(200, input_shape=(window_size, num_chars)))
    RNN_model.add(Dense(num_chars, activation='softmax'))
    
    return RNN_model
