# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:44:40 2018

#利用fasttext进行文本分类:
#处理语料,使之适合fasttext训练的格式

@author: Vino
"""
#import pandas as pd
import random
import fasttext
import os
import logging
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score

datafile = "datadir/dataSet.dat"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DIM_NUM=20
def transform_fasttext(dataSet,fasttextfile,column):
    labels = (dataSet['class']-1).tolist()
    texts = (dataSet[column]).tolist()
    string_list = []
    for i,(text,label) in enumerate(zip(texts,labels)):
        string = '__label__{} , {}'.format(label,text)
        string_list.append(string)
    write_data(string_list,fasttextfile)
    return True

def write_data(stringlist,savefile):
    with open(savefile,'w') as outfile:
        for string in stringlist:
            outfile.write(string+'\n')
    return True

def evaluate():
    return True

def predict_data(infile,testfile):
    string_list=[]
    with open(infile,'r') as fhandle:
        for i,line in enumerate(fhandle.readlines()):
            line = line.strip('\r\n')
            if i>0 and len(line)>0:
                strlist = line.split(',')
                string = '__label__{} , {}'.format(-1,strlist[2])
                #print(string)
                string_list.append(string)
    write_data(string_list,testfile)
    return True

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

def save_prob_file(test_id,probs,filename):
    #保存概率文件
    test_prob=pd.DataFrame(probs)
    test_prob.columns=["class_prob_%s"%i for i in range(1,probs.shape[1]+1)]
    test_prob['id']=list(test_id)
    test_prob.to_csv(filename,index=None)
    return True

def submit_result(model,f1,result_dir,predX):
    if os.path.exists(result_dir)==False:
        os.mkdir(result_dir)
    y_pred = model.predict(predX)
    probs = []
    prob_label = model.predict_proba(predX,k=19)
    for prob in prob_label:
        prob_temp = sorted(prob,key=lambda x:int(x[0]))
        arr = [p[1] for p in prob_temp]
        probs.append(arr)
    probs = np.array(probs)
    logging.info(probs.shape)
    logging.info("测试数据预测完成,开始写入结果文件")
    test_id = []
    with open("{}/lsi_{:.3f}.csv".format(result_dir,f1),'w') as outfile:
        outfile.write('id,class\n')
        for sid,pred in enumerate(y_pred):
            string = "{},{}".format(sid,int(pred[0])+1)
            test_id.append(sid)
            outfile.write(string+"\n")
    save_prob_file(test_id,probs,'{}/lsi_{:.3f}_prob.csv'.format(result_dir,f1))
    logging.info("数据保存完毕,请提交数据.")

if __name__ == "__main__":
    trainfile = 'train_set.csv'
    testfile = 'test_set.csv'
    column = 'word_seg'
    result_dir = "result"
    DIM_NUM=300
    (trainSet,valSet)=select_sample_by_class(trainfile,datafile)
    model_file = "model/word_seg_{}d.fasttext.bin".format(DIM_NUM)
    if os.path.exists(model_file)==False:
        logging.info("fasttext分类模型不存在,开始训练")
        transform_fasttext(trainSet,'datadir/train_fasttext.data',column)
        transform_fasttext(trainSet,'datadir/val_fasttext.data',column)
        logging.info("数据转化完成,开始训练fasttext分类模型")
        classify_model = fasttext.supervised("datadir/train_fasttext.data","model/word_seg_{}d.fasttext".format(DIM_NUM),lr=0.1,epoch=100,dim=DIM_NUM,bucket=50000000,loss='softmax',thread=56,min_count=3,word_ngrams=4,pretrained_vectors='fasttext.vec')
    else:
        logging.info("直接加载分类模型,对测试集进行预测")
        classify_model = fasttext.load_model('model/word_seg_{}d.fasttext.bin'.format(DIM_NUM), label_prefix='__label__')
    trainX,trainY = trainSet[column].tolist(),(trainSet['class']-1).tolist()
    y_pred = [int(pred[0]) for pred in classify_model.predict(trainX)]
    f2 = f1_score(trainY,y_pred,average='weighted')
    logging.info("训练集的f1分数:{}".format(f2))
    testX,testY = valSet[column].tolist(),(valSet['class']-1).tolist()
    predY = [int(pred[0]) for pred in classify_model.predict(testX)]
    f1 = f1_score(testY,predY,average='weighted')
    logging.info("验证集的f1分数:{}".format(f1))
    logging.info("模型训练完毕,对验证集进行测试")
    predX = pd.read_csv(testfile)[column].tolist()
    submit_result(classify_model,f1,result_dir,predX)
    logging.info("数据提交完毕,请查看:{}".format(result_dir))
