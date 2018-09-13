# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:29:33 2018
#用fasttext进行文本分类,并保存20w的特征到本地
#按类别比例选取95%的数据用来做训练集，并将id保存下来,方便下次:
@author: Vino
"""
import pandas as pd
import random
from collections import Counter
from gensim import models,corpora
import os
import numpy as np
import logging
from sklearn.metrics import f1_score
from sklearn.externals import joblib

datafile = "datadir/dataSet.dat"
NUM_DIM = 400
if os.path.exists("datadir")==False:
    os.mkdir("datadir")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dictionary = corpora.Dictionary.load("model/dict")

def select_sample_by_class(trainfile,idxfile):
    ratio = 0.7
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

def get_lsi_feature(dataSet,column,lsi):
    from gensim.matutils import corpus2dense
    corpus = dataSet[column].tolist()
    texts = [[word for word in doc.split(' ')] for doc in corpus]
    doc = [dictionary.doc2bow(text) for text in texts]
    lsi_vec = lsi[doc] #获取lsi向量
    features = corpus2dense(lsi_vec,NUM_DIM).T
    return features

def get_doc2vec_feature(dataSet,column,doc2vec):
    features = []
    corpus = dataSet[column].tolist()
    for lines in corpus:
        vector = doc2vec.infer_vector(lines)
        features.append(vector)
    return features

def load_doc2vec_feature(doc2vec_file):
    doc2vec_feature = joblib.load(doc2vec_file)
    return doc2vec_feature

def get_combine_feature(csvfile,column,doc2vec,lsi):
    if os.path.exists("datadir/train_feature_label.dat"):
        train_feature,train_label,val_feature,val_label = joblib.load("datadir/train_feature_label.dat")
        return train_feature,train_label,val_feature,val_label
    logging.info("加载训练集和验证集数据")
    (trainSet,valSet) = select_sample_by_class(csvfile,datafile)
    logging.info("提取训练集的lsi特征.")
    lsi_feature_train = get_lsi_feature(trainSet,column,lsi)
    logging.info("提取训练集的doc2vec特征")
    if os.path.exists("datadir/train_doc2vec.data"):
        logging.info("训练集的doc2vec特征,已训练好,直接加载.")
        doc2vec_feature_train = load_doc2vec_feature("datadir/train_doc2vec.data")
    else:
        doc2vec_feature_train = get_doc2vec_feature(trainSet,column,doc2vec)
        joblib.dump(doc2vec_feature_train,"datadir/train_doc2vec.data")
    logging.info("将训练集的doc2vec和lsi特征按列组合")
    train_feature = np.hstack([lsi_feature_train,doc2vec_feature_train])
    logging.info("训练集特征的维度:{}".format(train_feature.shape))
    train_label = (trainSet['class']-1).astype(int)
    logging.info("提取测试集的lsi特征.")
    lsi_feature_val = get_lsi_feature(valSet,column,lsi)
    logging.info("提取测试集的doc2vec特征.")
    if os.path.exists("datadir/val_doc2vec.data"):
        logging.info("测试集的doc2vec特征,已训练好,直接加载.")
        doc2vec_feature_val=load_doc2vec_feature("datadir/val_doc2vec.data")
    else:
        doc2vec_feature_val = get_doc2vec_feature(valSet,column,doc2vec)
        joblib.dump(doc2vec_feature_train,"datadir/val_doc2vec.data")
    logging.info("验证集中lsi_feature的维度:{}".format(np.array(lsi_feature_val).shape))
    logging.info("验证集中doc2vec的维度:{}".format(np.array(doc2vec_feature_val).shape))
    val_feature = np.hstack([lsi_feature_val,doc2vec_feature_val])
    logging.info("验证集特征的维度:{}".format(val_feature.shape))
    val_label = (valSet['class']-1).astype(int)
    return train_feature,train_label,val_feature,val_label

def train_classify(trainX,trainY,testX,testY,mode):
    from sklearn.model_selection import GridSearchCV
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    #trainX,testX,trainY,testY = train_test_split(dataX,dataY,test_size=0.3,random_state=33)
    if mode=="lightgbm":
        model = LGBMClassifier()
        model.fit(trainX,trainY)
        #clf = GridSearchCV(lgb_model,{'max_depth':[2,3,4]},cv=5,scoring='f1_weighted',verbose=1,n_jobs=-1)
    if mode=="SVC":
        from sklearn.svm import SVC
        reg = SVC(kernel='linear',verbose=True,probability=True)
        clf = GridSearchCV(reg,{'C':[0.1,1.0,10.0,100]},cv=5,scoring='f1_weighted',verbose=1,n_jobs=-1)
        clf.fit(trainX,trainY)
        logging.info(clf.best_score_)
        logging.info(clf.best_params_)
        model = clf.best_estimator_
    if mode=='LR':
        model = LogisticRegression(C=4, dual=True)
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

def predict_testSet(csvfile,doc2vec,lsi,column,classify_model):
    dataSet = pd.read_csv(csvfile)
    lsi_feature = get_lsi_feature(dataSet,column,lsi)
    if os.path.exists("datadir/test_doc2vec.data"):
        doc2vec_feature = joblib.load("datadir/test_doc2vec.data")
    else:
        doc2vec_feature = get_doc2vec_feature(dataSet,column,doc2vec)
    test_feature = np.hstack([lsi_feature,doc2vec_feature])
    predY = classify_model.predict(test_feature)
    predY_prob = classify_model.predict_prob(test_feature)
    return predY,predY_prob

def save_result(predY,predY_prob,f1,resultdir):
    if os.path.exists(resultdir)==False:
        os.mkdir(resultdir)
    with open(os.path.join(resultdir,"pred_{}.csv".format(f1))) as outfile:
        outfile.write('id,class\n')
        for i,pred in enumerate(predY):
            outfile.write("{},{}\n".format(i,pred))
    with open(os.path.join(resultdir,"prob_{}.csv".format(f1))) as outfile:
        outfile.write('id,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19\n')
        for i,line in enumerate(predY_prob):
            outfile.write("{}".format(i))
            for prob in line:
                outfile.write(',{}'.format(prob))
            outfile.write('\n')
    return True

if __name__ == "__main__":
    modeldir = "model"
    column = "word_seg"
    csvfile = "train_set.csv"
    testfile = "test_set.csv"
    mode = "LR"
    resultdir = "result"
    from gensim.models.doc2vec import Doc2Vec
    #(trainSet,valSet) = select_sample_by_class(csvfile,datafile)
    logging.info("开始对lsi和doc2vec的特征进行融合")
    lsi_file = os.path.join(modeldir,"lsi_{}_{}d.model".format(column,NUM_DIM))
    doc2vec_file = os.path.join(column,"doc2vec_{}d.model".format(NUM_DIM))
    doc2vec = Doc2Vec.load(doc2vec_file)
    lsimodel = models.LsiModel.load(lsi_file)
    logging.info("加载lsi和doc2vec模型完成,开始对特征进行融合")
    trainX,trainY,testX,testY = get_combine_feature(csvfile,column,doc2vec,lsimodel)
    logging.info("特征组合完毕,开始训练回归模型")
    classify_model,f1 = train_classify(trainX,trainY,testX,testY,mode)
    logging.info("模型训练完毕,开始对测试集进行预测")
    predY,predY_prob = predict_testSet(testfile,doc2vec,lsimodel,column,classify_model)
    logging.info("预测结果完毕,开始将结果写入指定文件夹")
    save_result(predY,predY_prob,f1,resultdir)
    logging.info("Done,请进行分析后提交")
