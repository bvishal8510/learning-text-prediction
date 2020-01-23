import numpy
import sys
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical
import tensorflow as tf
import utils
import codecs

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = codecs.open(filename, encoding = "utf8", errors ='replace').read()
raw_text = raw_text.lower()
# print(raw_text)
# types_of_encoding = ["utf8", "cp1252"]

# # create mapping of unique chars to integers
words = utils.preprocess(raw_text) #=raw_text
unique_words = sorted(list(set(words)))  #=chars

# print(words)
word_to_int, int_to_word = utils.create_lookup_tables(words)
# print(words_to_int)
# print(int_to_words)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# load the network weights
# filename = "weights-improvement-38-1.2788.hdf5"
filename = "weights-improvement-100-3.5546.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# # pick a random seed
# # start = numpy.random.randint(0, len(dataX)-1)
# # start = 86136
# # print("start",start)
# pattern = dataX[start]
# print("pattern",pattern)
# print ("Seed   :=", end='')
# # print(''.join([int_to_char[value] for value in pattern]))
# print([int_to_char[value] for value in pattern])

starting_words = input("Enter starting 5 words:").lower().split()
print(starting_words)
pattern = [word_to_int[word] for word in starting_words]
# print("pattern",pattern)

# generate characters
for i in range(30):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_word[index]
	sys.stdout.write(result)
	sys.stdout.write(' ')
	# print(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
