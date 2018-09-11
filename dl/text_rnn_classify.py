# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:52:12 2018

@author: Vino
"""
from mxnet import gluon,init
from mxnet.contrib import text
from utils import logging
import utils
import pandas as pd
from mxnet.gluon import data as gdata,loss as gloss


def main():
    num_outputs = 19
    lr = 0.1
    num_epochs = 100
    batch_size = 128
    embed_size = 300
    num_hiddens = 256
    num_layers = 2
    bidirectional = True
    ctx = utils.try_all_gpus()
    csvfile = "train_set.csv"
    column = 'word_seg'
    vocabfile = "{}.dict".format(column)
    vocab = utils.read_vocab(vocabfile)
    glove_embedding = text.embedding.CustomEmbedding(pretrained_file_path='{}.300d.txt'.format(column), vocabulary=vocab)
    net = utils.BiRNN(vocab, embed_size, num_hiddens, num_layers, bidirectional,
                num_outputs)
    net.initialize(init.Xavier(), ctx=ctx)
    # 设置 embedding 层的 weight 为预训练的词向量。
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    # 训练中不更新词向量（net.embedding 中的模型参数）。
    net.embedding.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainSet,valSet = utils.select_sample_by_class(csvfile,ratio=0.85)
    train_features,test_features,train_labels,test_labels=utils.read_dg_data(trainSet,valSet,vocab,column,MAX_LEN=2500)
    train_set = gdata.ArrayDataset(train_features, train_labels) #训练集
    test_set = gdata.ArrayDataset(test_features, test_labels) #测试集
    train_loader = gdata.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    logging.info("开始训练rnn {}文本分类模型".format(column))
    utils.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs,column)
    logging.info("模型训练完成,开始测试.")
    net.load_parameters('model/rnn_{}_best.param'.format(column),ctx=ctx)
    f1= utils.evaluate_valset(net,valSet,column)
    logging.info("rnn网络在验证集的f1_score:{}".format(f1))
    # net.save_parameters("model/rnn_{}_{:.4f}.param".format(column,f1))
    #--------------------------------------------------------------------------------
    logging.info("对数据进行测试")
    textSet = pd.read_csv('test_set.csv')
    y_probs,test_id = utils.predict_test_result(net,vocab,textSet,column,'result/rnn_{}_{:.4f}.csv'.format(column,f1))
    logging.info("保存概率数据")
    utils.save_prob_file(test_id,y_probs,'result/rnn_{}_{:.4f}_prob.csv'.format(column,f1))
    logging.info("保存完毕,请查看目录result.")

if __name__ == "__main__":
    main()
