# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:18:06 2018

@author: Administrator
"""
import numpy 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_int = dict((c,i) for i ,c in enumerate(alphabet))
int_to_char =dict((i,c) for i,c in enumerate(alphabet))

seq_lenth =3
dataX=[]
dataY=[]
pre=[]
for i in range(0,len(alphabet)-5-seq_lenth,1):
    seq_in = alphabet[i:i+seq_lenth]
    seq_out=alphabet[i+seq_lenth]
    dataX.append([char_to_int[char]for char in seq_in])
    dataY.append(char_to_int[seq_out])

for i in range(len(alphabet)-5,len(alphabet)-seq_lenth,1):
    seq_in = alphabet[i:i+seq_lenth]
    seq_out=alphabet[i+seq_lenth]
    pre.append([char_to_int[char]for char in seq_in])


X=numpy.reshape(dataX,(len(dataX),seq_lenth,1))
X=X/float(len(alphabet))
print(X)
Y = np_utils.to_categorical(dataY)


model=Sequential()
model.add(LSTM(32,input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(Y.shape[1],activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X, Y, nb_epoch=500, batch_size=1, verbose=2)
for pattern in pre:
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)