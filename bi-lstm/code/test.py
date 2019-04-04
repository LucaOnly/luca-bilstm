# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:02:54 2018

@author: Administrator
"""
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
import tensorflow as tf
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
import numpy as np
import pandas as pd
from keras.utils import np_utils
word_size = 128
maxlen = 32

data=[]
label=[]

def clean(s): #整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

#with open('D:\\NLP\\luca\\bi-lstm\\msr_train.txt') as f:
#    line = f.readline()
#    for k in range(30000):
#        line = f.readline()
#        line =clean(line)
#        line = re.split(u'[, 。！？、]/[bems]', line)
#        line = re.findall('(.)/(.)', str(line))
#        data.append(np.array(line)[:,0])
#        label.append(np.array(line)[:,1])
f=open('D:\\NLP\\luca\\bi-lstm\\msr_train.txt')
s = f.read()
s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})

d['label']=d['label'].apply(lambda x : np.array(list(x)+(['x']*(maxlen-len(x)))))

chars = [] #统计所有字，跟每个字编号
for i in data:
    chars.extend(i)
chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)
saveChars = dict(chars)
#保存chars
f = open('D:\\NLP\\luca\\bi-lstm\\code\\chars.txt','w')
f.write(str(saveChars)) 
f.close()  

d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
d['y'] =d['label'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5),tag[x].values.reshape(-1,1)))))

#d['y'] =d['label'].apply(lambda x: tag[x].values.reshape(-1,1))
#设计模型
word_size = 128
maxlen = 32
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1024
history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), 
batch_size=batch_size, 
nb_epoch=50)
model.save("D:\\NLP\\luca\\bi-lstm\\code\\modelSave\\test.h5")



