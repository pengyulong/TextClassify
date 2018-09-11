# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:55:53 2018
# 用gensim训练出来的doc2vec对文本进行分类:
@author: Vino
"""
import pandas as pd
import logging,os
from sklearn.metrics import f1_score
import numpy as np
from sklearn.externals import joblib
from collections import Counter
import random
from gensim import corpora,models
from collections import defaultdict
import sys
sys.path.insert(0,'..')
import utils

NUM_DIM = 50
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
datafile = "datadir/dataSet.dat"
#dictionary = corpora.Dictionary.load("model/{}.dict".format(column))
#lsi_model_dir = 'model/{}_{}_lsi.model'.format(column,NUM_DIM)

def get_lsi_feature(dataSet,column,lsi):
    from gensim.matutils import corpus2dense
    corpus = dataSet[column].tolist()
    texts = [[word for word in doc.split(' ')] for doc in corpus]
    doc = [dictionary.doc2bow(text) for text in texts]
    lsi_vec = lsi[doc] #获取lsi向量
    features = corpus2dense(lsi_vec,NUM_DIM).T
    return features


def process_corpus(docs,column):
    """
    --将输入为list的文本集合,保存为gensim能用的dictionary,corpus
    """
    dictionary=None,
    corpus = None
    dictfile = 'model/{}.dict'.format(column)
    corpusfile = 'model/{}.corpus'.format(column)
    if os.path.exists(dictfile)==True:
        logging.info("wait load corpus and dictionary ...")
        dictionary = corpora.Dictionary.load(dictfile)
        corpus = corpora.MmCorpus(corpusfile)
    else:
        logging.info("please wait create corpus and dictionary...")
        texts = [[word for word in document.split(' ')] for document in docs]
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token]+=1
        texts = [[token for token in text if frequency[token]>5] for text in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        dictionary.save(dictfile)
        corpora.MmCorpus.serialize(corpusfile,corpus)
    return corpus,dictionary

def train_lsi_model(corpus,dictionary,column):
    tfidf_file = 'model/{}_tfidf.model'.format(column)
    lsi_file = 'model/{}_{}_lsi.model'.format(column,NUM_DIM)
    if os.path.exists(tfidf_file):
        logging.info('load tfidf model...')
        tfidf = models.TfidfModel.load(tfidf_file)
    else:
        logging.info('train tfidf model...')
        tfidf = models.TfidfModel(corpus)
        tfidf.save(tfidf_file)
    if os.path.exists(lsi_file):
        logging.info('load lsi model...')
        lsi = models.LsiModel.load(lsi_file)
    else:
        logging.info("train lsi model...")
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=NUM_DIM)
        lsi.save(lsi_file)
    return tfidf,lsi



def save_prob_file(test_id,probs,filename):
    #保存概率文件
    test_prob=pd.DataFrame(probs)
    test_prob.columns=["class_prob_%s"%i for i in range(1,probs.shape[1]+1)]
    test_prob['id']=list(test_id)
    test_prob.to_csv(filename,index=None)
    return True

def load_data():
    if os.path.exists('datadir')==False:
        os.mkdir('datadir')
    if os.path.exists('model')==False:
        os.mkdir('model')
    for col in ['article','word_seg']:
        text1 = pd.read_csv('train_set.csv')[col].tolist()
        text2 = pd.read_csv('test_set.csv')[col].tolist()
        text1.extend(text2)
        corpus,dictionary = process_corpus(text1,column)
        tfidf,lsi = train_lsi_model(corpus,dictionary,column)
    return True

def submit_result(model,f1,result_dir,predX):
    if os.path.exists(result_dir)==False:
        os.mkdir(result_dir)
    y_pred = model.predict(predX)
    probs = model.pred_proba(predX)
    logging.info("测试数据预测完成,开始写入结果文件")
    test_id = []
    with open("{}/lsi_{:.3f}.csv".format(result_dir,f1),'w') as outfile:
        outfile.write('id,class\n')
        for (sid,pred) in zip(test_id,y_pred):
            string = "{},{}".format(sid,pred+1)
            test_id.append(sid)
            outfile.write(string+"\n")
    save_prob_file(test_id,probs,'{}/lsi_{:.3f}_prob.csv'.format(result_dir,f1))
    logging.info("数据保存完毕,请提交数据.")


if __name__ == "__main__":
    result_dir = 'result'
    column = ['word_seg','article']
    lsi_data = joblib.load('datadir/train_lsi_200d.dat')
    lda_data = joblib.load('datadir/train_lda_100d.dat')
    trainX_lsi,trainY,testX_lsi,testY = lsi_data
    logging.info("训练集lsi特征维度:{},测试集lsi特征维度:{}".format(trainX_lsi.shape,testX_lsi.shape))
    trainX_lda,_,testX_lda,_ = lda_data
    logging.info("训练集lda特征维度:{},测试集lda特征维度:{}".format(trainX_lda.shape,testX_lda.shape))
    trainX = np.hstack([trainX_lsi,trainX_lda])
    testX = np.hstack([testX_lsi,testX_lda])
    logging.info("trainX'shape:{},testX's shape:{}".format(trainX.shape,testX.shape))
    mode = 'lightgbm'
    model,f1=train_classify(trainX,trainY,testX,testY,mode)
    logging.info("模型训练完毕...")
    lsi_predX = joblib.load('datadir/predX_lsi_200d.dat')
    lda_predX = joblib.load('datadir/predX_lda_100d.dat')
    logging.info("predX_lsi'shape:{},predX_lda'shape:{}".format(lsi_predX.shape,lda_predX.shape))
    predX = np.hstack([lsi_predX,lda_predX])
    model,f1 = train_classify(trainX,trainY,testX,testY,mode)
    submit_result(model,f1,result_dir,predX)
    logging.info("模型训练完毕...")
