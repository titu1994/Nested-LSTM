'''Trains a Minimal RNN on the IMDB sentiment classification task.
The dataset is actually too small for Minimal RNN to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb

from nested_lstm import NestedLSTM

max_features = 20000
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# configuration matches 4.47 Million parameters with `units=600` and `64 embedding dim`
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(NestedLSTM(32, depth=2, dropout=0.0, recurrent_dropout=0.0))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint('weights/imdb_nlstm.h5', monitor='val_acc',
                                     save_best_only=True, save_weights_only=True)])

model.load_weights('weights/imdb_nlstm.h5')

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
