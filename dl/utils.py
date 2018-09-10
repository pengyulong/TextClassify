import collections
import time,random
from collections import Counter
import logging
import mxnet as mx
from mxnet.contrib import text
from mxnet import autograd, gluon,nd, init
from mxnet.gluon import data as gdata, utils as gutils,nn,rnn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def select_sample_by_class(trainfile,ratio):
    """
    将训练集的数据,按比例进行划分,得到训练集和验证集,用来调参
    """
    train_index,val_index = [],[]
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
    return (trainSet,valSet)

def accuracy(y_hat, y):
    """Get accuracy."""
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def count_tokens(samples):
    """Count tokens in the data set."""
    token_counter = collections.Counter()
    for sample in samples:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1
    return token_counter

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx,even_split=False),
            gutils.split_and_load(labels, ctx,even_split=False),
            features.shape[0])

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs,column):
    """Train and evaluate a model."""
    logging.info('training on', ctx)
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
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        if test_acc>=best_acc:
            best_acc = test_acc
            net.save_parameters("model/rnn_{}_best.param".format(column,best_acc))
        logging.info('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))
    logging.info("最优模型在测试集的acc:{}".format(best_acc))

def try_all_gpus():
    """Return all available GPUs, or [mx.cpu()] if there is no GPU."""
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

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

class TextCNN(nn.HybridBlock):
    def __init__(self, vocab, embedding_size, ngram_kernel_sizes,
                 nums_channels, num_outputs, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.ngram_kernel_sizes = ngram_kernel_sizes
        self.embedding_static = nn.Embedding(len(vocab), embedding_size)
        self.embedding_non_static = nn.Embedding(len(vocab), embedding_size)
        for i in range(len(ngram_kernel_sizes)):
            # 一维卷积层。
            conv = nn.Conv1D(nums_channels[i],
                             kernel_size=ngram_kernel_sizes[i], strides=1,
                             activation='relu')
            # 时序最大池化层。
            bn = nn.BatchNorm()
            pool = nn.GlobalMaxPool1D()
            # 将 self.conv_{i} 置为第 i 个 conv。
            setattr(self, 'conv_{i}', conv)
            setattr(self, 'bn_{i}', bn)
            # 将 self.pool_{i} 置为第 i 个 pool。
            setattr(self, 'pool_{i}', pool)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(num_outputs)

    def hybrid_forward(self, F, inputs):
        # 将 inputs 的形状由（批量大小，词数）变换为（词数，批量大小）。
        inputs = inputs.T
        # 根据 Conv1D 要求的输入形状，embeddings_static 和 embeddings_non_static
        # 的形状由（词数，批量大小，词向量维度）变换为（批量大小，词向量维度，词数）。
        embeddings_static = self.embedding_static(inputs).transpose((1, 2, 0))
        embeddings_non_static = self.embedding_non_static(
            inputs).transpose((1, 2, 0))
        # 将 embeddings_static 和 embeddings_non_static 按词向量维度连结。
        embeddings = F.concat(embeddings_static, embeddings_non_static,
                               dim=1)
        # 对于第 i 个卷积核，在时序最大池化后会得到一个形状为
        # （批量大小，nums_channels[i]，1）的矩阵。使用 flatten 函数将它形状压成
        # （批量大小，nums_channels[i]）。
        encoding = [
            F.flatten(self.get_pool(i)(self.get_bn(i)(self.get_conv(i)(embeddings))))
            for i in range(len(self.ngram_kernel_sizes))]
        # 将批量按各通道的输出连结。encoding 的形状：
        # （批量大小，nums_channels 各元素之和）。
        encoding = F.concat(*encoding, dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

    # 返回 self.conv_{i}。
    def get_conv(self, i):
        return getattr(self, 'conv_{i}')
    def get_bn(self, i):
        return getattr(self, 'bn_{i}')
    # 返回 self.pool_{i}。
    def get_pool(self, i):
        return getattr(self, 'pool_{i}')

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


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

def evaluate_valset(net,ValSet,vocab,column):
    logging.info("valSet 's shape:{}".format(ValSet.shape))
    docs = ValSet[column].tolist()
    y_true = ValSet['class'].tolist()
    y_pred = []
    for i,doc in enumerate(docs):
        sentence = nd.array([vocab.token_to_idx[token] for token in doc.split(' ')],ctx=try_gpu())
        output = net(nd.reshape(sentence,shape=(1,-1)))
        output = nd.softmax(output)
        label = int(nd.argmax(output,axis=1).asscalar()+1)
        y_pred.append(label)
    f1 = f1_score(y_true,y_pred,average='weighted')
    return f1

def predict_test_result(net,vocab,testSet,column,result_file):
    fhandle = open(result_file,'w')
    fhandle.write('id,class\n')
    logging.info("预测集的长度:{}".format(testSet.shape[0]))
    docs = testSet[column].tolist()
    y_probs,test_id = [],[]
    for i,doc in enumerate(docs):
        sentence = nd.array([vocab.token_to_idx[token] for token in doc.split(' ')],ctx=try_gpu())
        output = net(nd.reshape(sentence,shape=(1,-1)))
        prob = output.reshape(-1,).asnumpy().tolist()
        label = int(nd.argmax(output,axis=1).asscalar()+1)
        if i%9999==1:
            logging.info("the {}th document predict {}.".format(i,label))
        fhandle.write("{},{}\n".format(i,label))
        test_id.append(i)
        y_probs.append(prob)
    return y_probs,test_id