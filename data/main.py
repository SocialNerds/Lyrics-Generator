# -*- coding: utf-8 -*-
from __future__ import print_function
import helper
import numpy as np
import random
import sys

# How many characters we look back.
SEQUENCE_LENGTH = 80
# On how many characters we split a sequence.
SEQUENCE_STEP = 1
# The file that contains the text
CORPUS = "corpus.txt"
# How many epochs to train for.
EPOCHS = 7

# Get the text from corpus.
text = helper.read_corpus(CORPUS)
# Get unique characters.
chars = helper.extract_characters(text)

"""
    Create sequences that will be used as the input to the network.
    Create next_chars array that will serve as the labels during the training.
"""
# Create sequences that are the input values and the next characters that are the labels.
values, labels = helper.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)

char_to_index, indices_char = helper.get_chars_index_dicts(chars)

# Convert to vectors.
X, y = helper.vectorize(values, SEQUENCE_LENGTH, chars, char_to_index, labels)

# Create model.
model = helper.build_model(SEQUENCE_LENGTH, chars)

# Train the model and save it to the disk.
# model.fit(X, y, batch_size=512, epochs=EPOCHS)
# model.save_weights("final.h5")
# Uncomment the next line to use existing model weights. But you need to comment the two lines above.
# You need to have already run it at least once to save the first model weights.
model.load_weights("model_weights.h5")

# Create a first 80 chars seed.
seed = u"Του Κίτσου η μάνα με τα τέσσερα πουλάκια\nκαι με τα τρια τουφέκια στο βράχο απάνω".lower()

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    generated += seed

    print('----- Generating with seed: "' + seed + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(seed):
            x[0, t, char_to_index[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = helper.sample(predictions, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
