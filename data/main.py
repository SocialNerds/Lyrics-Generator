# -*- coding: utf-8 -*-
import helpers
import numpy as np
import random
import sys

# How many characters we look back.
SEQUENCE_LENGTH = 80
# On how many characters we split a sequence.
SEQUENCE_STEP = 1
# The file that contains the text.
CORPUS = "corpus.txt"
# How many epochs to train for.
EPOCHS = 10

# Get the text from corpus.
text = helpers.get_text(CORPUS)
# Get unique characters.
chars = helpers.get_unique_characters(text)
# Get length of unique chars.
chars_length = len(chars)

# Create sequences that are the input values and the next characters that are the labels.
values, labels = helpers.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)

char_to_index, index_to_char = helpers.create_dictionaries(chars)

# Convert to one hot arrays.
x, y = helpers.convert_to_one_hot(values, SEQUENCE_LENGTH, chars_length, char_to_index, labels)

# Create model.
model = helpers.create_model(SEQUENCE_LENGTH, chars_length)

# Train the model and save it to the disk.
# model.fit(x, y, batch_size=512, epochs=EPOCHS)
# model.save_weights("model_weights.h5")
# Uncomment the next line to use existing model weights. But you need to comment the two lines above.
# You need to have already run it at least once to save the first model weights.
model.load_weights("model_weights.h5")

# Create a first 80 chars seed.
print('_____________')
seed = u"Σύρε να ειπής της μάννας σου να μη σε καταρειέται\nνα πέσεις στο βουνό και να σου".lower()
sys.stdout.write(unicode(seed).encode('utf8'))
for i in range(400):
    x = np.zeros((1, SEQUENCE_LENGTH, chars_length))
    for t, char in enumerate(seed):
        x[0, t, char_to_index[char]] = 1.
    
    predictions = model.predict(x, verbose=0)[0]
    index = np.argmax(predictions)

    next_char = index_to_char[index]

    seed = seed[1:] + next_char

    sys.stdout.write(unicode(next_char).encode('utf8'))
    sys.stdout.flush()
print('\n_____________')