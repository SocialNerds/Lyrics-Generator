from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import io
import numpy as np

def create_sequences(text, sequence_length, step):
    """
    Create sequences that are the input values and the next characters that are the labels.

    Attributes
    ----------
    text : string
        The corpus that contains all the text.
    sequence_length : int
        How long each sequence will be.
    step : int
        How many steps each sequence will be apart.
    
    Returns
    -------
    list
        List of sequences.
    list
        List of labels, that are the next characters.
    """
    sequences = []
    labels = []
    for i in range(0, len(text) - sequence_length, step):
        sequences.append(text[i: i + sequence_length])
        labels.append(text[i + sequence_length])
    return sequences, labels


def create_model(sequence_length, vector_length):
    """
    Create the sequential model.

    Attributes
    ----------
    sequence_length : int
        How long each sequence will be.
    vector_length : int
        How long each character verctor will be.
    
    Returns
    -------
    Sequetial
        Keras sequential model.
    """
    model = Sequential()
    model.add(LSTM(1024, input_shape=(sequence_length, vector_length)))
    model.add(Dense(64))
    model.add(Dense(vector_length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    return model


def sample(preds, temperature=1.0):

    if temperature == 0:
        temperature = 1

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def get_unique_characters(text):
    """
    Get text unique characters.

    Attributes
    ----------
    text : string
        The full text
    
    Returns
    -------
    list
        A list with all the unique characters.
    """
    return sorted(list(set(text)))


def get_chars_index_dicts(chars):
    return dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars))


def get_text(path):
    """
    Get text of a file

    Attributes
    ----------
    path : string
        The file path
    
    Returns
    -------
    string
        File text.
    """
    with io.open(path, 'r', encoding='utf8') as f:
        return f.read().lower()

def vectorize(sequences, sequence_length, chars, char_to_index, next_chars):
    X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    return X, y

