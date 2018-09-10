# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:55:53 2018
# 用gensim提取语料的lda指定维度的lda特征,然后对文本进行分类:
@author: Vino
"""
import pandas as pd
import logging,os
from sklearn.metrics import f1_score
import numpy as np
from sklearn.externals import joblib
from collections import Counter
import random,warnings,time
from gensim import corpora,models
from collections import defaultdict
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
datafile = "datadir/dataSet.dat"
#dictionary = corpora.Dictionary.load("model/{}.dict".format(column))
#lsi_model_dir = 'model/{}_{}_lsi.model'.format(column,NUM_DIM)

def get_lda_feature(dataSet,column,lda):
    from gensim.matutils import corpus2dense
    corpus = dataSet[column].tolist()
    texts = [[word for word in doc.split(' ')] for doc in corpus]
    doc = [dictionary.doc2bow(text) for text in texts]
    lda_vec = lda[doc] #获取lsi向量
    features = corpus2dense(lda_vec,NUM_DIM).T
    logging.info("features' shape:{}".format(features.shape))
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

def train_lda_model(corpus,dictionary,column):
    tfidf_file = 'model/{}_tfidf.model'.format(column)
    lda_file = 'model/{}_{}_lda.model'.format(column,NUM_DIM)
    if os.path.exists(tfidf_file):
        logging.info('load tfidf model...')
        tfidf = models.TfidfModel.load(tfidf_file)
    else:
        logging.info('train tfidf model...')
        tfidf = models.TfidfModel(corpus)
        tfidf.save(tfidf_file)
    if os.path.exists(lda_file):
        logging.info('load lda model...')
        lda = models.LsiModel.load(lda_file)
    else:
        logging.info("train lda model...")
        corpus_tfidf = tfidf[corpus]
        lda = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=NUM_DIM,update_every=0,passes=20)
        lda.save(lda_file)
    return tfidf,lda

def select_sample_by_class(trainfile,idxfile):
    ratio = 0.85
    if os.path.exists(idxfile)==True:
        (trainSet,valSet) = joblib.load(idxfile)
        return trainSet,valSet
    else:
        train_index = []
        val_index = []
        df = pd.read_csv(trainfile)
        df.set_index('id',inplace=True)
        class_item = Counter(df['class'].tolist())
        for key in class_item.keys():
            temp = df[df['class']==key].index.tolist()
            random.shuffle(temp)
            nlength = len(temp)
            for i in range(nlength):
                if i < int(ratio*nlength):
                    train_index.append(temp[i])
                else:
                    val_index.append(temp[i])
        trainSet = df.loc[train_index]
        valSet = df.loc[val_index]
        joblib.dump((trainSet,valSet),idxfile)
        return (trainSet,valSet)

def train_classify(trainX,trainY,testX,testY,mode):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    #trainX,testX,trainY,testY = train_test_split(dataX,dataY,test_size=0.3,random_state=33)
    if mode=="lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(num_leaves=127)
        model.fit(trainX,trainY)
        #clf = GridSearchCV(lgb_model,{'max_depth':[2,3,4]},cv=5,scoring='f1_weighted',verbose=1,n_jobs=-1)
    if mode=="SVC":
        from sklearn.svm import SVC
        reg = SVC(kernel='linear',probability=True)
        clf = GridSearchCV(reg,{'C':[0.1,1.0,10.0,100]},cv=5,scoring='f1_weighted',verbose=1,n_jobs=-1)
        clf.fit(trainX,trainY)
        logging.info(clf.best_score_)
        logging.info(clf.best_params_)
        model = clf.best_estimator_
    if mode=='LR':
        reg = LogisticRegression(dual=True)
        clf = GridSearchCV(reg,{'C':[0.5,1,1.5,2]},cv=5,scoring='f1_weighted',verbose=1,n_jobs=-1)
        clf.fit(trainX,trainY)
        logging.info(clf.best_score_)
        logging.info(clf.best_params_)
        model = clf.best_estimator_
    y_pred1 = model.predict(testX)
    y_pred2 = model.predict(trainX)
    test_F1 = f1_score(testY,y_pred1,average='weighted')
    joblib.dump(model,"model/{}_{}.model".format(mode,test_F1))
    logging.info("测试集的f1分数:{}".format(test_F1))
    logging.info("训练集的f1分数:{}".format(f1_score(trainY,y_pred2,average='weighted')))
    return model,test_F1

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
        corpus,dictionary = process_corpus(text1,col)
        tfidf,lsi = train_lda_model(corpus,dictionary,col)
    return True

def submit_result(model,f1,result_dir,predX):
    if os.path.exists(result_dir)==False:
        os.mkdir(result_dir)
    y_pred = model.predict(predX)
    probs = model.predict_proba(predX)
    logging.info("测试数据预测完成,开始写入结果文件")
    test_id = []
    with open("{}/lsi_{:.3f}.csv".format(result_dir,f1),'w') as outfile:
        outfile.write('id,class\n')
        for sid,pred in enumerate(y_pred):
            string = "{},{}".format(sid,pred+1)
            test_id.append(sid)
            outfile.write(string+"\n")
    save_prob_file(test_id,probs,'{}/lsi_{:.3f}_prob.csv'.format(result_dir,f1))
    logging.info("数据保存完毕,请提交数据.")

if __name__ == "__main__":
    NUM_DIM = 100
    mode='lightgbm'
    column = ['word_seg','article']
    result_dir = 'result'
    t1 = time.time()
    load_data()
    train_data_file = 'datadir/train_lda_{}d.dat'.format(NUM_DIM)
    predX_file = 'datadir/predX_lda_{}d.dat'.format(NUM_DIM)
    try:
        train_data = joblib.load(train_data_file)
        predX = joblib.load(predX_file)
        trainX,trainY,testX,testY = train_data
    except Exception as err:
        logging.info(err)
        train_file = 'train_set.csv'
        test_file = 'test_set.csv'
        trainSet,valSet=select_sample_by_class(train_file,datafile)
        testSet = pd.read_csv(test_file)
        train_features = []
        val_features = []
        test_features = []
        for col in column:
            lsi = models.LsiModel.load('model/{}_{}_lda.model'.format(col,NUM_DIM))
            dictionary = corpora.Dictionary.load('model/{}.dict'.format(col))
            train_features.append(get_lda_feature(trainSet,col,lsi))
            val_features.append(get_lda_feature(valSet,col,lsi))
            test_features.append(get_lda_feature(testSet,col,lsi))
        logging.info("开始组合特征....")
        trainX = np.hstack(train_features)
        testX = np.hstack(val_features)
        predX = np.hstack(test_features)
        trainY = (trainSet['class']-1).astype(int)
        testY = (valSet['class']-1).astype(int)
        train_data = (trainX,trainY,testX,testY)
        joblib.dump(train_data,'datadir/train_lda_{}d.dat'.format(NUM_DIM))
        joblib.dump(predX,'datadir/test_lda_{}.dat'.format(NUM_DIM))
    logging.info("训练{}分类模型".format(mode))
    model,f1 = train_classify(trainX,trainY,testX,testY,mode)
    submit_result(model,f1,result_dir,predX)
    t2= time.time()
    print("cost time:{}".format(t2-t1))