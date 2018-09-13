# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:52:12 2018

@author: Vino
"""
from mxnet import gluon,init
from mxnet.contrib import text
import sys
sys.path.insert(0,'..')
from utils import logging
import utils
import pandas as pd
from mxnet.gluon import data as gdata,loss as gloss
from Parameter import RNNParameter


def main(column,DIM_NUM):
    column = 'word_seg'
    NUM_DIM = 300
    Params = RNNParameter(column,NUM_DIM)
    num_outputs = Params.num_outputs
    lr = Params.lr
    num_epochs = Params.num_epochs
    batch_size = Params.batch_size
    embed_size = DIM_NUM
    num_hiddens = Params.num_hiddens
    num_layers = Params.num_layers
    bidirectional = Params.bidirectional
    ctx = utils.try_all_gpus()
    csvfile = Params.train_file
    vocab = utils.read_vocab(Params.vocab_file)
    glove_embedding = text.embedding.CustomEmbedding(pretrained_file_path=Params.embedding_file, vocabulary=vocab)
    net = utils.BiRNN(vocab, embed_size, num_hiddens, num_layers, bidirectional,
                num_outputs)
    net.initialize(init.Xavier(), ctx=ctx)
    # 设置 embedding 层的 weight 为预训练的词向量。
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    # 训练中不更新词向量（net.embedding 中的模型参数）。
    net.embedding.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainSet,valSet = utils.select_sample_by_class(csvfile,ratio=0.85)
    train_features,test_features,train_labels,test_labels=utils.read_dg_data(trainSet,valSet,vocab,column,MAX_LEN=2500)
    train_set = gdata.ArrayDataset(train_features, train_labels) #训练集
    test_set = gdata.ArrayDataset(test_features, test_labels) #测试集
    train_loader = gdata.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    logging.info("开始训练rnn {}文本分类模型".format(column))
    best_acc = utils.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs,column,Params.best_param_file)
    logging.info("模型训练完成,最佳模型的acc:{} 开始测试.".format(best_acc))
    net.load_parameters(Params.best_param_file,ctx=ctx)
    f1= utils.evaluate_valset(net,valSet,vocab,column)
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
    main('word_seg',300)
