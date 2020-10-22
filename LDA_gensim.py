import os
import re
import jieba
import jieba.posseg as pseg 
from gensim.models import LdaModel
from gensim.corpora import Dictionary  
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.externals import joblib
# from sklearn.decomposition import LatentDirichletAllocation

current_path = os.getcwd()
train_file_path = current_path + r'\article\cnews.train.txt'
test_file_path = current_path + r'\article\cnews.test.txt'
file_done_path = current_path + r'\final.txt'
stop_words1_path = current_path + r'\stopwords-master\stopwords-master\baidu_stopwords.txt'
stop_words2_path = current_path + r'\stopwords-master\stopwords-master\cn_stopwords.txt'
stop_words3_path = current_path + r'\stopwords-master\stopwords-master\hit_stopwords.txt'
stop_words4_path = current_path + r'\stopwords-master\stopwords-master\scu_stopwords.txt'

class TestProcess():

    def __init__(self,train_file_path,test_file_path,file_done_path):
        self.train_file_path = train_file_path
        # self.filestoppath = filestoppath
        self.test_file_path = test_file_path
        self.file_done_path = file_done_path
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
                # line_seg = seg_depart(line.strip())
                self.file_train_read.append(line)
                count += 1
        print("训练文本共有%d行"%count)
        # print(self.fileTrainRead)
        return self.file_train_read

    #加载停用词表
    def load_stopwords(self):
        stop_words1 = [line.rstrip() for line in open(stop_words1_path, 'r', encoding='utf-8')]
        stop_words2 = [line.rstrip() for line in open(stop_words2_path, 'r', encoding='utf-8')]
        stop_words3 = [line.rstrip() for line in open(stop_words3_path, 'r', encoding='utf-8')]
        stop_words4 = [line.rstrip() for line in open(stop_words4_path, 'r', encoding='utf-8')]
        sw = stop_words1 + stop_words2 + stop_words3 +stop_words4
        # print(sw)
        return sw

    #对文本进行分词去停用词
    def segLine(self):
        for row_line in (self.file_train_read):
            sentence_seged = jieba.cut(row_line.strip(),cut_all=False)         
            stop_words = self.load_stopwords()
            new_list = [word for word in sentence_seged if word not in stop_words]
            self.word_list_train_seg.append(new_list)
        print("分词以及去除停用词完成")
        # # print(self.word_list_train_seg)
        # for i in range(len(self.word_list_train_seg)):
        #     print(self.word_list_train_seg[i])
        return self.word_list_train_seg

    #处理后写入文件
    def write_word_text(self):
        with open(self.file_done_path,'wb') as f:
            for i in range(len(self.word_list_train_seg)):
                f.write(self.word_list_train_seg[i])
                f.write('\n'.encode('utf-8'))

    #lda模型 自定义10个主题
    def lda_module(self):
        dictionary = Dictionary(self.word_list_train_seg)
        corpus = [dictionary.doc2bow(text) for text in self.word_list_train_seg]
        lda = LdaModel(corpus=corpus,id2word=dictionary,num_topics=10)
        print(lda.print_topics(10))
        return lda

    #测试文档处理
    def save_test_line(self):
        count = 0
        with open(self.test_file_path,encoding='utf-8') as file_test_row:
            for index,line in enumerate (file_test_row):
                line = line.split('\t')[1]
                line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
                self.file_test_read.append(line)
                count += 1
        print("测试文本共有%d行"%count)
        # print(self.fileTrainRead)
        return self.file_test_read

    #测试文档分词以及去除停用词处理
    def segLine_test(self):
        for row_line in (self.file_test_read):
            sentence_seged = jieba.cut(row_line.strip(),cut_all=False)         
            stop_words = self.load_stopwords()
            new_list = [word for word in sentence_seged if word not in stop_words]
            self.word_list_test_seg.append(new_list)
        print("分词以及去除停用词完成")
        return self.word_list_test_seg

    #测试模型
    def lda_test_module(self):
        dictionary = Dictionary(self.word_list_test_seg)
        corpus_test = [dictionary.doc2bow(text) for text in self.word_list_test_seg]
        topics_test = tp.lda_module().get_document_topics(corpus_test)
        labels = ['体育','娱乐','科技']
        for i in range(3):
            print('这条'+labels[i]+'新闻的主题分布为：\n')
            print(topics_test[i],'\n')

if __name__ == '__main__':

    tp = TestProcess(train_file_path,test_file_path,file_done_path)
    tp.saveLine()
    tp.load_stopwords()
    tp.segLine()
    tp.write_word_text()
    # tp.lda_module()
    # tp.save_test_line()
    # tp.segLine_test()
    # tp.lda_test_module()