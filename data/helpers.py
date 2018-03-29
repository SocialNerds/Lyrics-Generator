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

def create_model(sequence_length, chars_length):
    """
    Create the sequential model.

    Attributes
    ----------
    sequence_length : int
        How long each sequence will be.
    chars_length : int
        How long each character vector will be.
    
    Returns
    -------
    Sequetial
        Keras sequential model.
    """
    model = Sequential()
    model.add(LSTM(1024, input_shape=(sequence_length, chars_length)))
    model.add(Dense(64))
    model.add(Dense(chars_length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    return model

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

def create_dictionaries(chars):
    """
    Returns two arrays as dictioanries, one as values/keys and one as keys/values.

    Attributes
    ----------
    chars : list
        List of unique characters.
    
    Returns
    -------
    list
        List as values/keys.
    list
        List as keys/values.
    """
    return dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars))

def convert_to_one_hot(sequences, sequence_length, chars_length, char_to_index, labels):
    """
    Convert sequences and labels to one hot arrays.

    Attributes
    ----------
    sequences : list
        List that contains the sequences of the characters.
    sequence_length : int
        The length of each sequence.
    chars_length : int
        Length of the array of the unique characters.
    char_to_index : list
        A list that is a dictionary with the unique characters and indexes.
    labels : list
        List of labels.

    Returns
    -------
    list
        List of values as sequences of one hot arrays.
    list
        List of labels as one hot arrays.
    """
    x = np.zeros((len(sequences), sequence_length, chars_length), dtype=np.bool)
    y = np.zeros((len(sequences), chars_length), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_index[char]] = 1
        y[i, char_to_index[labels[i]]] = 1

    return x, y
