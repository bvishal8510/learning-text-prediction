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


#reading preprocessed data
processed_data_file = open('processed-data','rb')

word_to_int = pickle.load(processed_data_file)
int_to_word = pickle.load(processed_data_file)
seq_length = pickle.load(processed_data_file)
n_vocab = pickle.load(processed_data_file)
len_sentences = pickle.load(processed_data_file)
dataX = pickle.load(processed_data_file)
dataY = pickle.load(processed_data_file)
processed_data_file.close()

n_patterns = len(dataX)
# print ("Total Patterns: ", n_patterns)  

# reshape X to be [samples, time steps, features] reshape(array, shape, order)
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# print(X)

# normalize            #check here dividing by len(sentences)
X = X / float(n_vocab)
# print(X)

# one hot encode the output variable #converts array into multiple arrays with corresponding value as 1 rest as 0
y = to_categorical(dataY)
# print(y)
# print(X.shape)
# print(y.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))       #check the significance of input_shape try with X.shape[0], X.shape[1]
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))      #try using linear or ReLU   
model.compile(loss='categorical_crossentropy', optimizer='adam')   #mean squared error

# load the network weights
# filename = "weights-improvement-38-1.2788.hdf5"
filename = "weights-improvement-298-4.2165.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


print("Enter starting ",seq_length," words:")
starting_words = input().lower().split()
# print(starting_words)
pattern = [word_to_int[word] for word in starting_words]
# print("pattern",pattern)

# generate characters
for i in range(1):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	print(x)
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_word[index]
	sys.stdout.write(result)
	sys.stdout.write(' ')
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print()
