# -*- coding: utf-8 -*-
"""
@author: pengyulong
"""
import fasttext
import os
import pandas as pd
from sklearn.metrics import f1_score
from .utils import logging,transform_fasttext,select_sample_by_class,write_data,save_prob_file
from .Parameter import FasttextParameter


class FasttextClassify(FasttextParameter):
    def __init__(self,column,fasttext_dim):
        FasttextParameter.__init__(self,column,fasttext_dim)
        self.model_file = os.path.join(self.model_dir,"{}_{}d.fasttext.bin".format(self.column,self.fasttext_dim))
        self.preTrained_vectors = os.path.join(self.model_dir,"fasttext_{}_{}d.vec".format(self.column,self.fasttext_dim))

    def load_data(self):
        logging.info("加载数据与模型....")
        (self.trainSet,self.valSet)= select_sample_by_class(self.train_file,ratio=0.85)
        transform_fasttext(self.trainSet,self.fasttext_train_file,self.column)
        transform_fasttext(self.valSet,self.fasttext_val_file,self.column)
        if os.path.exists(self.model_file):
            logging.info("fasttext分类模型已存在,直接加载模型...")
            self.model = fasttext.load_model(self.model_file, label_prefix=self.prefix_label)
        else:
            logging.info("fasttext分类模型不存在,开始重新训练...")
            self.model = self.train_model()
        return True

    def train_model(self):
        classify_model=None
        if os.path.exists(self.preTrained_vectors):
            logging.info("存在预训练的词向量,从本地加载词向量进行训练...")
            classify_model = fasttext.supervised(self.fasttext_train_file,self.model_file[0:-4],
                                                 lr=0.1,epoch=100,dim=self.fasttext_dim,bucket=50000000,
                                                 loss='softmax',thread=56,min_count=3,word_ngrams=4,
                                                 pretrained_vectors=self.preTrained_vectors)
            self.best_score=self.evaluate()
        else:
            logging.info("不存在预训练的词向量,重头开始训练...")
            classify_model = fasttext.supervised(self.fasttext_train_file, self.model_file[0:-4], lr=0.1, epoch=100,
                                                 dim=self.fasttext_dim, bucket=50000000, loss='softmax', thread=56,
                                                 min_count=3, word_ngrams=4)
            self.best_score=self.evaluate()
        return classify_model

    def predict(self):
        logging.info("对测试数据进行预测...")
        predX = pd.read_csv(self.test_file)[self.column].tolist()
        y_pred = self.model.predict(predX)
        probs_list = self.model.predict_proba(predX,self.num_outputs)
        y_probs,result_string,i= [],['id,class\n'],0
        logging.info("预测结果完毕,开始将预测结果和类别概率写入文件...")
        for (prob,label) in zip(probs_list,y_pred):
            probs = sorted(prob,key=lambda x:int(x[0]))
            arr = [p[1] for p in probs]
            y_probs.append(arr)
            result_string.append("{},{}\n".format(i,int(label[0])+1))
            i = i+1
        result_submit = os.path.join(self.result_dir,"fasttext_{}_{}.csv".format(self.column,self.best_score))
        result_probs = os.path.join(self.result_dir,"fasttext_prob_{}_{:.4f}.csv".format(self.column,self.best_score))
        save_prob_file(y_probs,result_probs)
        write_data("".join(result_string),result_submit)
        logging.info("数据提交完毕,请查看:{}".format(self.result_dir))

    def evaluate(self):
        logging.info("开始验证模型在验证集上的准确率...")
        trainX,trainY = self.trainSet[self.column].tolist(),(self.trainSet['class']-1).tolist()
        valX,valY = self.valSet[self.column].tolist(),(self.valSet['class']-1).tolist()
        y_pred_train = [int(pred[0]) for pred in self.model.predict(trainX)]
        y_pred_val = [int(pred[0]) for pred in self.model.predict(valX)]
        f1 = f1_score(trainY, y_pred_train, average='macro')
        f2 = f1_score(valY,y_pred_val,average='macro')
        logging.info("模型在训练集的f1:{:.4f}".format(f1))
        logging.info("模型在验证集的f1:{:.4f}".format(f2))
        return f2

if __name__ == "__main__":
    fasttext_classify = FasttextClassify("word_seg",20)
    fasttext_classify.load_data()
    fasttext_classify.predict()

