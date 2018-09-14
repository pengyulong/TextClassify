# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:55:53 2018
# 用gensim提取语料的lda指定维度的lda特征,然后对文本进行分类:
@author: Vino
"""
import pandas as pd
import os
from sklearn.metrics import f1_score
import numpy as np
from sklearn.externals import joblib
from collections import Counter
import random,warnings,time
from gensim import corpora,models
from collections import defaultdict
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from utils import logging,train_classify,write_data,save_prob_file
from Parameter import ProjectPath

class LDAClassify(ProjectPath):
    def __init__(self,num_topics,column,docs,mode):
        ProjectPath.__init__(self)
        self.num_topics = num_topics
        self.column = column
        self.docs = docs
        self.mode = mode
        self.lda_model_file = os.path.join(self.model_dir,"lda_{}_{}d.model".format(self.column,self.num_topics))
        self.lda_data_file = os.path.join(self.data_dir,"lda_{}_{}.dat".format(self.column,self.num_topics))
        # self.tidif_model_file = os.path.join(self.model_dir,"tfidf_{}.model".format(self.column))
        self.dict_file = os.path.join(self.model_dir,"{}.dict".format(column))
        self.corpus_file = os.path.join(self.model_dir,"{}.corpus".format(column))

    def load_data(self):
        '''
        通过对输入的corpus进行加工,得到gensim能用的dictionary和corpus,用于后续训练lda模型
        加载数据的训练集trainSet和预测集testSet
        :return: no
        '''
        if os.path.exists(self.dict_file) == True:
            logging.info("wait load corpus and dictionary ...")
            dictionary = corpora.Dictionary.load(self.dict_file)
            corpus = corpora.MmCorpus(self.corpus_file)
            self.corpus = corpus
            self.dictionary = dictionary
        else:
            logging.info("please wait create corpus and dictionary...")
            texts = [[word for word in document.split(' ')] for document in self.docs]
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            texts = [[token for token in text if frequency[token] > 5] for text in texts]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            dictionary.save(self.dict_file)
            corpora.MmCorpus.serialize(self.corpus_file,corpus)
            self.corpus = corpus
            self.dictionary = dictionary


    def train_lda_model(self):
        '''
        训练lda模型,或者加载lda模型
        :return: lda模型
        '''
        if os.path.exists(self.lda_model_file):
            logging.info('load lda model...')
            lda = models.LsiModel.load(self.lda_model_file)
        else:
            logging.info("train lda model...")
            #corpus_tfidf = tfidf[corpus]
            lda = models.LdaModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics, update_every=0, passes=20)
            lda.save(self.lda_model_file)
        self.model = lda

    def _get_lda_feature(self,corpus):
        '''
        :param 对任意的list类型的语料,提取对应模型的特征,用于训练分类模型
        :return: numpy或scipy的features
        '''
        from gensim.matutils import corpus2dense
        texts = [[word for word in doc.split(' ')]for doc in corpus]
        doc = [self.dictionary.doc2bow(text) for text in texts]
        lda_vec = self.model[doc]
        features = corpus2dense(lda_vec,self.num_topics).T
        return features

    def train_classify(self):
        '''
        通过机器学习,训练分类模型
        :return:
        '''
        dataX,dataY = self._get_lda_feature(self.trainSet[self.column].tolist()),(self.trainSet['class']-1).tolist()
        self.classify_model,self.best_score = train_classify(dataX,dataY,self.mode)

    def save_features(self):
        '''
        保存对训练集和预测集提取的features用于后续调节分类模型或者特征的concat操作
        :return:
        '''
        if os.path.exists(self.lda_data_file):
            (trainX,testX)=joblib.load(self.lda_data_file)
        else:
            trainX,testX = self._get_lda_feature(self.trainSet[self.column].tolist()),self._get_lda_feature(self.testSet[self.column].tolist())
            joblib.dump((trainX,testX),self.lda_data_file)
        self.testX = testX

    def predict(self):
        '''
        将预测集的数据通过分类器进行预测，并得到类别结果和概率
        :return:
        '''
        y_pred = self.classify_model.predict(self.testX)
        y_prob = self.classify_model.predict_proba(self.testX)
        logging.info("测试数据预测完成,开始写入结果文件...")
        result_file = os.path.join(self.result_dir,"lda_{}_{}_{}d_{:.3f}.csv".format(self.column,self.mode,self.num_topics,self.best_score))
        result_prob_file = os.path.join(self.result_dir,"lda_{}_{}_{}d_{:.3f}_prob.csv".format(self.column,self.mode,self.num_topics,self.best_score))
        result_string = ['id,class\n']
        for id,pred in enumerate(y_pred):
            string = "{},{}\n".format(id,pred+1)
            result_string.append(string)
        write_data(result_string,result_file)
        save_prob_file(y_prob,result_prob_file)
        logging.info("数据提交完毕,请查看{}...".format(self.result_dir))
        return True


if __name__ == "__main__":
    df1 = pd.read_csv('../data/train_set.csv')
    df2 = pd.read_csv('../data/test_set.csv')
    column = 'word_seg'
    text1 = set(df1[column].tolist())
    text2 = set(df2[column].tolist())
    corpus = list(text1 | text2)
    num_topics = 100
    logging.info("load corpus succed...")
    mode = "lightgbm"
    lda_classify = LDAClassify(num_topics,column,corpus,mode)
    lda_classify.load_data()
    lda_classify.train_lda_model()
    lda_classify.save_features()
    lda_classify.train_classify()
    lda_classify.predict()


