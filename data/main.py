# -*- coding: utf-8 -*-
import helpers
import numpy as np
import random
import sys
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder

# How many characters we look back.
SEQUENCE_LENGTH = 80
# On how many characters we split a sequence.
SEQUENCE_STEP = 1
# The file that contains the text.
CORPUS = "corpus.txt"
# How many epochs to train for.
EPOCHS = 3

# Get the text from corpus.
text = helpers.get_text(CORPUS)
# Get unique characters.
chars = helpers.get_unique_characters(text)

# Create sequences that are the input values and the next characters that are the labels.
values, labels = helpers.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)

char_to_index, indices_char = helpers.get_chars_index_dicts(chars)

# Convert to vectors.
X, y = helpers.vectorize(values, SEQUENCE_LENGTH, chars, char_to_index, labels)

# tokenizer = Tokenizer(char_level=True, filters='')
# tokenizer.fit_on_texts(text)
# print(tokenizer.word_index)
# tokenized = tokenizer.texts_to_sequences(values[-5:])
# print(tokenized)
# enc = OneHotEncoder()
# enc.fit(tokenized)
# oh = enc.transform(tokenized).toarray()
# print(oh)
# print(len(oh[0]))
# exit()

# Create model.
model = helpers.create_model(SEQUENCE_LENGTH, len(chars))

# Train the model and save it to the disk.
model.fit(X, y, batch_size=512, epochs=EPOCHS)
model.save_weights("model_weights.h5")
# Uncomment the next line to use existing model weights. But you need to comment the two lines above.
# You need to have already run it at least once to save the first model weights.
# model.load_weights("model_weights.h5")

# Create a first 80 chars seed.
seed = u"Του Κίτσου η μάνα με τα τέσσερα πουλάκια\nκαι με τα τρια τουφέκια στο βράχο απάνω".lower()

# diversities = [0.2, 0.5, 1.0, 1.2]
diversities = [0.2]
for diversity in diversities:
    print('----- diversity:', diversity)

    generated = ''
    generated += seed
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(seed):
            x[0, t, char_to_index[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
    
        # max_value = max(predictions)
        # max_index = predictions.index(max_value)
        print(np.argmax(predictions))

        next_index = helpers.sample(predictions, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        seed = seed[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
