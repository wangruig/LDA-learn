'''
Descripttion: 
version: Python 3.6.3
Author: 王瑞国
Date: 2020-09-27 10:31:22
LastEditors: 王瑞国
LastEditTime: 2020-10-16 21:01:36
'''
import os
import re
import jieba

import jieba.posseg as pseg 
from multiprocessing import Pool
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation


current_path = os.getcwd()
train_file_path = current_path + r'\article\cnews.train.txt'
# train_file_path = current_path +r'\article\cnews.test1.txt'
test_file_path = current_path + r'\article\cnews.test.txt'
# file_done_path = current_path + r'final.txt'
stop_words1_path = current_path + r'\stopwords-master\stopwords-master\baidu_stopwords.txt'
stop_words2_path = current_path + r'\stopwords-master\stopwords-master\cn_stopwords.txt'
stop_words3_path = current_path + r'\stopwords-master\stopwords-master\hit_stopwords.txt'
stop_words4_path = current_path + r'\stopwords-master\stopwords-master\scu_stopwords.txt'


class TestProcess():
 # Windows后台如何运行程序 nohup linux下命令！
    def __init__(self,train_file_path,test_file_path):
        self.train_file_path = train_file_path
        # self.filestoppath = filestoppath
        self.test_file_path = test_file_path
        # self.file_done_path = file_done_path
        self.file_train_read = [] #初始文本保存的所有行列表
        self.word_list_train_seg = [] #分词以及去除停用词所保存的列表
        self.file_test_read = [] #测试分词后保存的所有行列表
        self.word_list_test_seg = [] #测试分词以及去除停用词所保存的列表

    #读取文本并保存到列表中
    def saveLine(self):
        count = 0
        with open(self.train_file_path,encoding='utf-8') as file_train_row:
            for index,line in enumerate (file_train_row):
                line = line.split('\t')[1]
                line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
                self.file_train_read.append(line)
                count += 1
        print("训练文本共有%d行"%count)
        return self.file_train_read

    #加载停用词表
    def load_stopwords(self):
        stop_words1 = [line.rstrip() for line in open(stop_words1_path, 'r', encoding='utf-8')]
        stop_words2 = [line.rstrip() for line in open(stop_words2_path, 'r', encoding='utf-8')]
        stop_words3 = [line.rstrip() for line in open(stop_words3_path, 'r', encoding='utf-8')]
        stop_words4 = [line.rstrip() for line in open(stop_words4_path, 'r', encoding='utf-8')]
        sw = stop_words1 + stop_words2 + stop_words3 +stop_words4
        return sw
    #多线程
    def sub_process(self, word_list, stop_word_list):
        result = ''
        for word in word_list:

            if word not in stop_word_list:
                result += str(word) + ' '

        return result[:-1]
    
    def get_process(self,mutil_word_list, stop_word_list):
        pool = Pool(processes=3)

        sub_process_map = partial(self.sub_process, stop_word_list=stop_word_list)
        res = pool.map(sub_process_map, mutil_word_list)
        pool.close()
        pool.join()
        # print(res)
        return res
    #对文本进行分词去停用词
    def segLine(self):
        aa = []
        for row_line in (self.file_train_read):
            sentence_seged = jieba.cut(row_line.strip(),cut_all=False)
            sentence_seged_result = ' '.join(sentence_seged)

            sentence_seged_result = sentence_seged_result.split(' ')
            aa.append(sentence_seged_result)

        stop_words = self.load_stopwords()
        str1 = self.get_process(aa,stop_words)

        print("分词以及去除停用词完成")
        return str1

    #处理后写入文件
    def write_word_text(self):
        with open(self.file_done_path,'wb') as f:
            for i in range(len(self.word_list_train_seg)):
                f.write(self.word_list_train_seg[i][0].encode('utf-8'))
                f.write('\n'.encode('utf-8'))

    #转换为词频向量
    def count_vector(self):
        corpus = self.segLine()
        # stop_word = self.load_stopwords()
        cont_vector = CountVectorizer()
        contf = cont_vector.fit_transform(corpus)
        return contf,cont_vector
    #调用sklearn的lda模型
    def lda_module(self):
        n_top_words = 20    
        cont_f,cont_vector = self.count_vector() # 为啥(5,31)维度??to do cont_f.shape()=(5000,83437)
        lda = LatentDirichletAllocation(n_topics=10,max_iter=10,verbose=-1,learning_method='batch',random_state=42)

        docres = lda.fit_transform(cont_f) # fit和fit_transform的区别？？

        feature_names = cont_vector.get_feature_names()
        # joblib.dump(lda,'lda_module.jl')
        doc_topics_list = lda.transform(cont_f[0:2]) #doc_topics_list.shape(2，10)
        print(doc_topics_list)   
        for topic_idx, topic in enumerate(lda.components_):
            print ("Topic #%d:" % topic_idx)
            print (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print(lda.perplexity(cont_f))
        print(docres)
        print(lda.components_)
        return lda

    def saveLine_test(self):
        count = 0
        with open(self.test_file_path,encoding='utf-8') as file_test_row:
            for index,line in enumerate (file_test_row):
                line = line.split('\t')[1]
                line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
                self.file_test_read.append(line)
                count += 1
        print("训练文本共有%d行"%count)
        return self.file_test_read 
    
    #对文本进行分词去停用词
    def segLine_test(self):
        aa = []
        for row_line in (self.file_test_read):
            sentence_seged = jieba.cut(row_line.strip(),cut_all=False)
            sentence_seged_result = ' '.join(sentence_seged)

            sentence_seged_result = sentence_seged_result.split(' ')
            aa.append(sentence_seged_result)

        stop_words = self.load_stopwords()
        str1 = self.get_process(aa,stop_words)

        print("分词以及去除停用词完成")
        return str1
   
 
    # def count_vector_test(self):
    #     corpus = self.segLine_test()
    #     # stop_word = self.load_stopwords()
    #     cont_vector = CountVectorizer(max_df=0.95,min_df=2)
    #     contf = cont_vector.fit_transform(corpus) # contf.shape()=(3,4)
    #     return contf,cont_vector
    
    def load_lda(self):
        corpus = self.segLine_test()
        # stop_word = self.load_stopwords()
        cont_vector = CountVectorizer()
        contf = cont_vector.fit_transform(corpus) # contf.shape()=(3,339)
        lda_module = joblib.load('lda_module.jl')
        
if __name__ == '__main__':

    tp = TestProcess(train_file_path,test_file_path)
    tp.saveLine()
    # tp.load_stopwords()
    tp.lda_module()
    tp.saveLine_test()

 

  




