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


# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = codecs.open(filename, encoding = "utf8", errors ='replace').read()
raw_text = raw_text.lower()
# print(raw_text)

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
	sentence_words = sentence.split()
	if(len(sentence_words)>seq_length):
		for i in range(0, len(sentence_words) - seq_length):
			seq_in = sentence_words[i:i + seq_length]
			seq_out = sentence_words[i + seq_length]
			dataX.append([word_to_int[word] for word in seq_in])
			dataY.append(word_to_int[seq_out])


n_patterns = len(dataX)
# print ("Total Patterns: ", n_patterns)  

# reshape X to be [samples, time steps, features] reshape(array, shape, order)
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize          #check here dividing by len(sentences)
X = X / float(n_vocab)
# print(X)

# one hot encode the output variable #converts array into multiple arrays with corresponding value as 1 rest as 0
y = to_categorical(dataY)
# print(y)

# # print(y)
# # print(X.shape)
# print(y.shape)

# #writing processed data to file
processed_data_file = open('processed-data','wb')

pickle.dump(word_to_int, processed_data_file)
pickle.dump(int_to_word, processed_data_file)
pickle.dump(seq_length, processed_data_file)
pickle.dump(n_vocab, processed_data_file)
processed_data_file.close()

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))       #check the significance of input_shape try with X.shape[0], X.shape[1]
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
model.add(Dense(256, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# to train the model only run once else regret
# model.fit(X, y, epochs=150, batch_size=128, callbacks=callbacks_list)
