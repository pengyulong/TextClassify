# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:52:12 2018

@author: Vino
"""
import sys
sys.path.insert(0,'..')
import gluonbook as gb
from mxnet import gluon,init,nd,autograd
from mxnet.contrib import text
from mxnet.gluon import data as gdata,loss as gloss,utils as gutils,nn,rnn
import random,logging,os,time
import pandas as pd
import collections
from sklearn.externals import joblib
from collections import Counter
import mxnet as mx
from sklearn.metrics import f1_score
datafile = "datadir/dataSet.dat"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx,even_split=False),
            gutils.split_and_load(labels,ctx,even_split=False),
            features.shape[0])

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs,column):
    """Train and evaluate a model."""
    print('training on', ctx)
    best_acc = 0.7
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0
        start = time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx)
        if test_acc>=best_acc:
            best_acc = test_acc
            net.save_parameters("model/rnn_{}_best.param".format(column,best_acc))
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))
    logging.info("最优模型在测试集的acc:{}".format(best_acc))

def select_sample_by_class(trainfile,idxfile):
    ratio = 0.9
    if False:#os.path.exists(idxfile)==True:
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

class BiRNN(nn.Block):
    """
    定义双向lstm模型
    """
    def __init__(self,vocab,embed_size,num_hiddens,num_layers,bidirectional,num_outputs,**kwargs):
        super(BiRNN,self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab),embed_size)
        self.encoder = rnn.LSTM(num_hiddens,num_layers=num_layers,bidirectional=bidirectional,input_size=embed_size)
        self.decoder = nn.Dense(num_outputs,flatten=False)
    def forward(self,inputs):
        embeddings = self.embedding(inputs.T)
        states = self.encoder(embeddings)
        encoding = nd.concat(states[0],states[-1])
        outputs = self.decoder(encoding)
        return outputs

def read_dg_data(trainSet,valSet,vocab,column):
    """
    读取竞赛方提供的数据,并进行预处理,作为text rnn模型的输入
    """
    def encode_samples(token_samples,vocab):
        features = []
        for sample in token_samples:
            feature = []
            for token in sample.split(' '):
                if token in vocab.token_to_idx:
                    feature.append(vocab.token_to_idx[token])
                else:
                    feature.append(0)
            features.append(feature)
        return features

    def pad_sample(features,maxlen=1000,PAD=0):
        padded_features = []
        for feature in features:
            if len(feature)> maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                PAD = random.sample(feature,1)[0]
                while len(padded_feature)<maxlen:
                    padded_feature.append(PAD)
            padded_features.append(padded_feature)
        return padded_features
    train_features = encode_samples(trainSet[column],vocab)
    test_features = encode_samples(valSet[column],vocab)
    train_features = nd.array(pad_sample(train_features))
    test_features = nd.array(pad_sample(test_features))
    train_labels = nd.array((trainSet['class']-1).astype(int))
    test_labels = nd.array((valSet['class']-1).astype(int))
    return train_features,test_features,train_labels,test_labels

def save_prob_file(test_id,probs,filename):
    #保存概率文件
    test_prob=pd.DataFrame(probs)
    num_outputs = len(probs[0])
    test_prob.columns=["class_prob_%s"%i for i in range(1,num_outputs+1)]
    test_prob['id']=list(test_id)
    test_prob.to_csv(filename,index=None)
    return True

def read_vocab(vocabfile):
    word_count = collections.Counter()
    with open(vocabfile,'r') as infile:
        for line in infile.readlines():
            line = line.strip('\r\n')
            [word,count] = line.split(' ')
            word_count[word]=int(count)
    vocab = text.vocab.Vocabulary(word_count,unknown_token='<unk>',reserved_tokens=None)
    return vocab

def evaluate_cnn_f1(net,ValSet,column):
    logging.info("valSet 's shape:{}".format(ValSet.shape))
    docs = ValSet[column].tolist()
    y_true = ValSet['class'].tolist()
    y_pred = []
    for i,doc in enumerate(docs):
        sentence = nd.array([vocab.token_to_idx[token] for token in doc.split(' ')],ctx=gb.try_gpu())
        output = net(nd.reshape(sentence,shape=(1,-1)))
        output = output.exp()/output.exp().sum(axis=1)
        label = int(nd.argmax(output,axis=1).asscalar()+1)
        y_pred.append(label)
    f1 = f1_score(y_true,y_pred,average='weighted')
    return f1


def predict_text_cnn(net,vocab,testSet,column,result_file):
    fhandle = open(result_file,'w')
    fhandle.write('id,class\n')
    logging.info("预测集的长度:{}".format(testSet.shape[0]))
    docs = testSet[column].tolist()
    y_probs,test_id = [],[]
    for i,doc in enumerate(docs):
        sentence = nd.array([vocab.token_to_idx[token] for token in doc.split(' ')],ctx=gb.try_gpu())
        output = net(nd.reshape(sentence,shape=(1,-1)))
        prob = output.reshape(-1,).asnumpy().tolist()
        label = int(nd.argmax(output,axis=1).asscalar()+1)
        if i%9999==1:
            logging.info("the {}th document predict {}.".format(i,label))
        fhandle.write("{},{}\n".format(i,label))
        test_id.append(i)
        y_probs.append(prob)
    return y_probs,test_id

if __name__ == "__main__":
    num_outputs = 19
    lr = 0.1
    num_epochs = 100
    batch_size = 128
    embed_size = 300
    num_hiddens = 256
    num_layers = 2
    bidirectional = True
    ctx = gb.try_all_gpus()
    csvfile = "train_set.csv"
    column = 'word_seg'
    vocabfile = "{}.dict".format(column)
    vocab = read_vocab(vocabfile)
    glove_embedding = text.embedding.CustomEmbedding(pretrained_file_path='{}.300d.txt'.format(column), vocabulary=vocab)
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers, bidirectional,
                num_outputs)
    net.initialize(init.Xavier(), ctx=ctx)
    # 设置 embedding 层的 weight 为预训练的词向量。
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    # 训练中不更新词向量（net.embedding 中的模型参数）。
    net.embedding.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainSet,valSet = select_sample_by_class(csvfile,datafile)
    train_features,test_features,train_labels,test_labels=read_dg_data(trainSet,valSet,vocab,column)
    train_set = gdata.ArrayDataset(train_features, train_labels) #训练集
    test_set = gdata.ArrayDataset(test_features, test_labels) #测试集
    train_loader = gdata.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    logging.info("开始训练rnn {}文本分类模型".format(column))
    train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs,column)
    logging.info("模型训练完成,开始测试.")
    net.load_parameters('model/rnn_{}_best.param'.format(column),ctx=ctx)
    f1= evaluate_cnn_f1(net,valSet,column)
    logging.info("rnn网络在验证集的f1_score:{}".format(f1))
    # net.save_parameters("model/rnn_{}_{:.4f}.param".format(column,f1))
    #--------------------------------------------------------------------------------
    logging.info("对数据进行测试")
    textSet = pd.read_csv('test_set.csv')
    y_probs,test_id = predict_text_cnn(net,vocab,textSet,column,'result/rnn_{}_{:.4f}.csv'.format(column,f1))
    logging.info("保存概率数据")
    save_prob_file(test_id,y_probs,'result/rnn_{}_{:.4f}_prob.csv'.format(column,f1))
    logging.info("保存完毕,请查看目录result.")