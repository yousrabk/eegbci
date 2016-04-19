"""
    @author : stan
    goal : cnn for dreem challenge
"""

# my imports
from local import path

# general import
import h5py
import numpy as np

# sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# keras imports
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization


f = h5py.File(path + "data_dreem_challenge.hdf5", "r")

X = f["eeg_train"][:]
y = f["stages_train"][:]

lb = LabelBinarizer()
y = lb.fit_transform(y)

nb_filter = 15
filter_length = 10
init = "glorot_uniform"
border_mode = "valid"

model = Sequential()

# model.add(BatchNormalization())
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        input_shape=(X.shape[1], 1),
                        activation="relu",
                        init=init,
                        border_mode=border_mode,
                        ))

model.add(MaxPooling1D(pool_length=2))

# model.add(BatchNormalization())
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        activation="relu",
                        init=init,
                        border_mode=border_mode))

model.add(MaxPooling1D(pool_length=2))

# model.add(BatchNormalization())
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        activation="relu",
                        init=init,
                        border_mode=border_mode))

model.add(MaxPooling1D(pool_length=2))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation("relu"))

# 5 is the number of classes
model.add(Dense(5,
                activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model.fit(X_train, y_train, nb_epoch=10,
          validation_split=0.5,
          batch_size=512,
          verbose=1)
