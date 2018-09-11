
from mxnet import gluon,init
from mxnet.contrib import text
from mxnet.gluon import data as gdata,loss as gloss
import pandas as pd
import sys
sys.path.insert(0,'..')
import utils
from utils import logging

def main():
    DIM_NUM=300
    num_outputs = 19
    lr = 0.01
    num_epochs = 1
    batch_size = 64
    embed_size = 300
    ngram_kernel_sizes = [3, 4, 5]
    nums_channels = [100, 100, 100]
    ctx = utils.try_all_gpus()
    column = 'word_seg'
    csvfile = "../data/train_set.csv"
    vocabfile = "../data/{}.dict".format(column)
    vocab = utils.read_vocab(vocabfile)
    glove_embedding = text.embedding.CustomEmbedding(pretrained_file_path='../data/{}.300d.txt'.format(column), vocabulary=vocab)
    net = utils.TextCNN(vocab, embed_size, ngram_kernel_sizes, nums_channels,num_outputs)
    #net.hybridize()
    net.initialize(init.Xavier(), ctx=ctx)
    # embedding_static 和 embedding_non_static 均使用预训练的词向量。
    net.embedding_static.weight.set_data(glove_embedding.idx_to_vec)
    #net.embedding_non_static.weight.set_data(glove_embedding.idx_to_vec)
    # 训练中不更新 embedding_static 的词向量，即不更新 embedding_static 的模型参数。
    net.embedding_static.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainSet,valSet = utils.select_sample_by_class(csvfile,ratio=0.85)
    train_features,test_features,train_labels,test_labels=utils.read_dg_data(trainSet,valSet,vocab,column,MAX_LEN=2500)
    train_set = gdata.ArrayDataset(train_features, train_labels) #训练集
    test_set = gdata.ArrayDataset(test_features, test_labels) #测试集
    train_loader = gdata.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    logging.info("开始训练cnn {} 文本分类模型".format(column))
    utils.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs,column)
    logging.info("模型训练完成,开始测试.")
    f1= utils.evaluate_valSet(net,valSet,column)
    logging.info("cnn网络在验证集的f1_score:{}".format(f1))
    try:
        net.load_parameters('model/rnn_{}_best.param'.format(column),ctx=ctx)
    except Exception as err:
        logging.info("模型精度不够,请重新设置参数")
    f1= utils.evaluate_valSet(net,valSet,column)
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
