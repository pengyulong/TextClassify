
from mxnet import gluon,init
from mxnet.contrib import text
from mxnet.gluon import data as gdata,loss as gloss
import pandas as pd
import os
import utils
from utils import logging
from Parameter import CNNParameter

def main(column,DIM_NUM):
    '''
    :params column 表示使用数据集中的哪部分语料
    :params DIM_NUM 表示使用的词向量的维度
    '''
    Params = CNNParameter(column,DIM_NUM)
    Params.load_dir()
    ctx = utils.try_all_gpus()
    csvfile = Params.train_file
    vocabfile = Params.vocab_file
    vocab = utils.read_vocab(vocabfile)
    glove_embedding = text.embedding.CustomEmbedding(pretrained_file_path=Params.embedding_file, vocabulary=vocab)
    net = utils.TextCNN(vocab,DIM_NUM, Params.ngram_kernel_sizes, Params.nums_channels,Params.num_outputs)
    net.initialize(init.Xavier(), ctx=ctx)
    # embedding_static 和 embedding_non_static 均使用预训练的词向量。
    net.embedding_static.weight.set_data(glove_embedding.idx_to_vec)
    #net.embedding_non_static.weight.set_data(glove_embedding.idx_to_vec)
    # 训练中不更新 embedding_static 的词向量，即不更新 embedding_static 的模型参数。
    net.embedding_static.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': Params.lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainSet,valSet = utils.select_sample_by_class(csvfile,ratio=0.85)
    train_features,test_features,train_labels,test_labels=utils.read_dg_data(trainSet,valSet,vocab,column,MAX_LEN=2500)
    train_set = gdata.ArrayDataset(train_features, train_labels) #训练集
    test_set = gdata.ArrayDataset(test_features, test_labels) #测试集
    train_loader = gdata.DataLoader(train_set, batch_size=Params.batch_size,shuffle=True)
    test_loader = gdata.DataLoader(test_set, batch_size=Params.batch_size, shuffle=False)
    logging.info("开始训练cnn {} 文本分类模型...".format(column))
    best_acc = utils.train(train_loader, test_loader, net, loss, trainer, ctx,Params.num_epochs,column,Params.best_param_file)
    logging.info("模型训练完成,最佳的acc:{:.4f} 开始测试...".format(best_acc))
    net.load_parameters(Params.best_param_file,ctx=ctx)
    f1= utils.evaluate_valset(net,valSet,vocab,column)
    best_file = os.path.join(Params.result_dir,"rnn_{}_{:.4f}.csv".format(column,f1))
    best_prob_file = os.path.join(Params.result_dir,"rnn_{}_{:.4f}_prob.csv".format(column,f1))
    logging.info("cnn网络在验证集最佳的f1_score:{:.4f}".format(f1))
    logging.info("对数据进行测试")
    textSet = pd.read_csv(Params.test_file)
    y_probs = utils.predict_test_result(net,vocab,textSet,column,best_file)
    logging.info("保存概率数据...")
    utils.save_prob_file(y_probs,best_prob_file)
    logging.info("保存完毕,请查看目录result.")

if __name__ == "__main__":
    main('word_seg',300)
