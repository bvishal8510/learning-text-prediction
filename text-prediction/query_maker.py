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
import pickle

# processed_data_file = open('processed-data','rb')

# word_to_int = pickle.load(processed_data_file)
# print(word_to_int)
# int_to_word = pickle.load(processed_data_file)
# print(int_to_word)
# seq_length = pickle.load(processed_data_file)
# print(seq_length)
# X = pickle.load(processed_data_file)
# print(X)
# y = pickle.load(processed_data_file)
# print(y)
# # seq_length = int(processed_data_file.readline())
# # print(type(int_to_word))
# # X = list(processed_data_file.readline())
# # print(type(int_to_word))
# # y = list(processed_data_file.readline())
# # print(type(int_to_word))
# processed_data_file.close()


# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = codecs.open(filename, encoding = "utf8", errors ='replace').read()
raw_text = raw_text.lower()
# print(raw_text)
# types_of_encoding = ["utf8", "cp1252"]

# create mapping of unique chars to integers
words, sentences = utils.preprocess(raw_text)

unique_words = sorted(list(set(words)))

# print(words)
word_to_int, int_to_word = utils.create_lookup_tables(unique_words)
# print(words_to_int)
# print(int_to_words)


n_vocab = len(unique_words)   
# print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []

for sentence in sentences:
	# print(sentence)
	sentence_words = sentence.split()
	if(len(sentence_words)>seq_length):
		for i in range(0, len(sentence_words) - seq_length):
			seq_in = sentence_words[i:i + seq_length]
			seq_out = sentence_words[i + seq_length]
			dataX.append([word_to_int[word] for word in seq_in])
			# print(seq_out)
			dataY.append(word_to_int[seq_out])


n_patterns = len(dataX)
# print ("Total Patterns: ", n_patterns)  
# #50 - 90024
# #10 - 90064
# #5 - 90069

# reshape X to be [samples, time steps, features] reshape(array, shape, order)
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# print(X)

# normalize
X = X / float(n_vocab)
# print(X)
# one hot encode the output variable #converts array into multiple arrays with corresponding value as 1 rest as 0
y = to_categorical(dataY)
# print(y)
# print(X.shape)
# print(y.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# load the network weights
# filename = "weights-improvement-38-1.2788.hdf5"
filename = "weights-improvement-298-4.2165.hdf5"
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
print("Enter starting ",seq_length," words:")
starting_words = input().lower().split()
print(starting_words)
pattern = [word_to_int[word] for word in starting_words]
# print("pattern",pattern)

# generate characters
for i in range(5):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_word[index]
	sys.stdout.write(result)
	sys.stdout.write(' ')
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
