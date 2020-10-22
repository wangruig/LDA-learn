'''
Descripttion: 
version: Python 3.6.3
Author: 王瑞国
Date: 2020-09-24 10:32:52
LastEditors: 王瑞国
LastEditTime: 2020-10-22 15:51:57
'''
import os
import re
import jieba
import pickle
import numpy as np
import pandas as pd
import jieba.posseg as pseg 
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

current_path = os.getcwd()
# train_file_path = current_path + r'\Data\cnews.train.txt'
train_file_path = current_path +r'\Data\cnews.test1.txt'
test_file_path = current_path + r'\Data\cnews.test.txt'
stop_words1_path = current_path + r'\Data\baidu_stopwords.txt'
stop_words2_path = current_path + r'\Data\cn_stopwords.txt'
stop_words3_path = current_path + r'\Data\hit_stopwords.txt'
stop_words4_path = current_path + r'\Data\scu_stopwords.txt'

# 读取文件保存为文本列表
def saveLine(train_file_path):
    file_train_read = []
    count = 0
    with open(train_file_path,encoding='utf-8') as file_train_row:
        for index,line in enumerate (file_train_row):
            line = line.split('\t')[1]
            line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
            file_train_read.append(line)
            count += 1
    print("训练文本共有%d行"%count)
    return file_train_read

# 加载停用词表
def load_stopwords():
    stop_words1 = [line.rstrip() for line in open(stop_words1_path, 'r', encoding='utf-8')]
    stop_words2 = [line.rstrip() for line in open(stop_words2_path, 'r', encoding='utf-8')]
    stop_words3 = [line.rstrip() for line in open(stop_words3_path, 'r', encoding='utf-8')]
    stop_words4 = [line.rstrip() for line in open(stop_words4_path, 'r', encoding='utf-8')]
    sw = stop_words1 + stop_words2 + stop_words3 +stop_words4
    return sw

# 多线程
def sub_process(word_list, stop_word_list):
    result = ''
    for word in word_list:
        if word not in stop_word_list:
            result += str(word) + ' '
    return result[:-1]

def get_process(mutil_word_list, stop_word_list):
    pool = Pool(processes=3)
    sub_process_map = partial(sub_process, stop_word_list=stop_word_list)
    res = pool.map(sub_process_map, mutil_word_list)
    pool.close()
    pool.join()
    # print(res)
    return res

# 对文本进行分词并去停用词处理
def segLine():
    aa = []
    file_train_read = saveLine(train_file_path)
    for row_line in file_train_read:
        sentence_seged = jieba.cut(row_line.strip(),cut_all=False)
        sentence_seged_result = ' '.join(sentence_seged)
        sentence_seged_result = sentence_seged_result.split(' ')
        aa.append(sentence_seged_result)

    stop_words = load_stopwords()
    str1 = get_process(aa,stop_words)

    print("分词以及去除停用词完成")
    return str1


#转换为词频向量
def count_vector(corpus):
    
    cont_vector = CountVectorizer()
    contf = cont_vector.fit_transform(corpus)
    return contf,cont_vector,cont_vector.vocabulary_

def count_vector_test(corpus):
    
    vocabulary_load = pickle.load(open(current_path+r'\Model\temp_cont_vector_V.pkl','rb'))
    cont_vector = CountVectorizer(vocabulary=vocabulary_load)
    contf = cont_vector.transform(corpus)
    return contf,cont_vector

# def lda_model(contf,cont_vector,n_topics,contf_test,cont_vector_test):

def lda_model(contf,cont_vector,n_topics):
    '''
    desc : 进行lda模型的训练
    params ：
        contf ：文本分词后词频向量
        cont_vector : 
        n_topics : 主题个数
    return : lda模型
    '''
    # df_keys_words = pd.Data
    n_top_words = 20
    lda = LatentDirichletAllocation(n_topics=n_topics,max_iter=30,learning_method='online', learning_offset=50.,random_state=0,verbose=-1)
    docres = lda.fit_transform(contf)
    feature_names = cont_vector.get_feature_names()
    # best_model = model.best_estimator_
    # print('Best Model params:',model.best_params_)
    # print('Best log likelihood score:',model.best_score_)
    keywords_topics = []
    for topic_idx, topic in enumerate(lda.components_):
        # result = []
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        result = [" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])]
        keywords_topics.append(result)
    pickle.dump(keywords_topics,open(current_path+r'\Model\temp_keywords_topics.pkl','wb'))
    print(lda.perplexity(contf))
    # print(lda_output)
    print(lda.components_)
    pickle.dump(lda, open(current_path+r'\Model\lda.pkl','wb'))
    return lda
   
def main():
    # corpus = segLine()
    # 训练保存模型变量
    # pickle.dump(corpus,open(current_path+r'\Model\temp_corpus.pkl','wb'))
    # contf,cont_vector,cont_vector_V = count_vector(corpus)
    # pickle.dump(contf,open(current_path+r'\Model\temp_contf.pkl','wb'))
    # pickle.dump(cont_vector,open(current_path+r'\Model\temp_cont_vector.pkl','wb'))
    # pickle.dump(cont_vector_V,open(current_path+r'\Model\temp_cont_vector_V.pkl','wb'))
    # cont_vector = pickle.load(open(current_path+r'\Model\temp_cont_vector.pkl','rb'))
    # contf = pickle.load(open(current_path+r'\Model\temp_contf.pkl','rb'))
    # lda_model(contf,cont_vector,10)




    # 测试保存模型变量
    # pickle.dump(corpus,open(current_path+r'\Model\temp_corpus_test.pkl','wb'))
    # contf,cont_vector = count_vector_test(corpus)
    # pickle.dump(contf,open(current_path+r'\Model\temp_contf_test.pkl','wb'))
    # pickle.dump(cont_vector,open(current_path+r'\Model\temp_cont_vector_test.pkl','wb'))
    
    
    # corpus_test = pickle.load(open(current_path+r'\Model\temp_corpus_test.pkl','rb'))
    # contf_test = pickle.load(open(current_path+r'\Model\temp_contf_test.pkl','rb'))

    # cont_vector_test = pickle.load(open(current_path+r'\Model\temp_cont_vector_test.pkl','rb'))
    lda = pickle.load(open(current_path+r'\Model\lda.pkl','rb'))
    # show_topics(cont_vector,lda,20)
    # topic_probability_score = lda.transform(contf_test)
    # print(topic_probability_score)
    # print(np.argmax(topic_probability_score))
    # print(1)
    






if __name__ == '__main__':
    main()
    