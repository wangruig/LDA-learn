'''
Descripttion: 
version: Python 3.6.3
Author: 王瑞国
Date: 2020-10-19 21:42:34
LastEditors: 王瑞国
LastEditTime: 2020-10-22 17:10:08
'''

import pickle
from stop_words import lda_model
import os
import numpy as np


if __name__ == '__main__':

    current_path = os.getcwd()
    lda = pickle.load(open(current_path+r'\Model\lda.pkl','rb'))
    contf_test = pickle.load(open(current_path+r'\Model\temp_contf_test.pkl','rb'))
    keywords_topics = pickle.load(open(current_path+r'\Model\temp_keywords_topics.pkl','rb'))
    topic_probability_score =np.matrix(lda.transform(contf_test))
    topic_doc_lsit = topic_probability_score/topic_probability_score.sum(axis=1)
    for i in range(len(keywords_topics)):
        print(keywords_topics[i])
    pridict_topic = topic_doc_lsit.argmax(axis=1).tolist()
    for j in range(len(pridict_topic)):
        print('第%d行的文档主题是%d'%(j+1,pridict_topic[j][0]+1))