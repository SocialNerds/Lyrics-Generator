# -*- coding: utf-8 -*-
"""
    Inspired by https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""

from __future__ import print_function
import helper
import numpy as np
import random
import sys
from keras.models import load_model

"""
    Define global variables.
"""
SEQUENCE_LENGTH = 80
SEQUENCE_STEP = 1
PATH_TO_CORPUS = "corpus.txt"
EPOCHS = 7
DIVERSITY = 1.0

"""
    Read the corpus and get unique characters from the corpus.
"""
text = helper.read_corpus(PATH_TO_CORPUS)
chars = helper.extract_characters(text)

"""
    Create sequences that will be used as the input to the network.
    Create next_chars array that will serve as the labels during the training.
"""
sequences, next_chars = helper.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)
char_to_index, indices_char = helper.get_chars_index_dicts(chars)

"""
    The network is not able to work with characters and strings, we need to vectorise.
"""
X, y = helper.vectorize(sequences, SEQUENCE_LENGTH, chars, char_to_index, next_chars)

"""
    Define the structure of the model.
"""
model = helper.build_model(SEQUENCE_LENGTH, chars)

"""
    Train the model
"""

# model.fit(X, y, batch_size=512, epochs=EPOCHS)
# model.save_weights("final.h5")
model.load_weights("final_1024_64_7.h5")

"""
    Pick a random sequence and make the network continue
"""


for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    # insert your 40-chars long string. OBS it needs to be exactly 40 chars!
    # sentence = u"Του Κίτσου η μάνα με τα τέσσερα πουλάκια"
    sentence = u"Του Κίτσου η μάνα με τα τέσσερα πουλάκια\nκαι με τα τρια τουφέκια στο βράχο απάνω"
    sentence = sentence.lower()
    generated += sentence

    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = helper.sample(predictions, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()



