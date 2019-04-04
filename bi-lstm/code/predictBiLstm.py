# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:03:37 2018

@author: Administrator
"""
import re
from keras.models import load_model
import numpy as np
word_size = 128
maxlen = 32
model=load_model("D:\\NLP\\luca\\bi-lstm\\code\\modelSave\\test.h5")  


f = open('D:\\NLP\\luca\\bi-lstm\\code\\chars.txt','r')  
chars = f.read()
f.close()   
chars = eval(chars)
#分词
#转移概率，单纯用了等概率
zy = {'be':0.5, 
      'bm':0.5, 
      'eb':0.5, 
      'es':0.5, 
      'me':0.5, 
      'mm':0.5,
      'sb':0.5, 
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}
def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            keylist = list(nows.keys())
            valuelist = list(nows.values())
            paths[keylist[k]] =valuelist[k]
    pathkeys = list(paths.keys())
    pathValues = list(paths.values())
    return pathkeys[np.argmax(pathValues)]

def simple_cut(s):
    if s:
        tt = list(s)
        result =[]
        for x in tt:
            if x in chars:
                result.append(chars[x])
            else:
                result.append(0)
        result = result+[0]*(maxlen-len(tt))
        r = model.predict(np.array(result).reshape(-1,32), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []
    
not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')

def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result

print(cut_word("魏则西是一名大学生"))