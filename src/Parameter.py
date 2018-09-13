import os

class ProjectPath(object):
    def __init__(self):
        self.data_dir = '../data'
        self.model_dir = '../model'
        self.result_dir = '../result'
        self.train_file = os.path.join(self.data_dir,"train_set.csv")
        self.test_file = os.path.join(self.data_dir,"test_set.csv")
    def load_dir(self):
        if os.path.exists(self.data_dir)==False:
            os.makedirs(self.data_dir)
        if os.path.exists(self.model_dir)==False:
            os.makedirs(self.model_dir)
        if os.path.exists(self.result_dir)==False:
            os.makedirs(self.result_dir)
    
class CNNParameter(ProjectPath):
    def __init__(self,column,DIM_NUM):
        ProjectPath.__init__(self)
        self.num_outputs = 19
        self.num_epochs = 1
        self.lr = 0.01
        self.batch_size = 64
        self.ngram_kernel_sizes = [3,4,5]
        self.nums_channels = [100,100,100]
        self.vocab_file = os.path.join(self.data_dir,"{}.dict".format(column))
        self.embedding_file = os.path.join(self.data_dir,"{}.{}d.txt".format(column,DIM_NUM))
        self.best_param_file = os.path.join(self.model_dir,"cnn_{}_best.param".format(column))
        

class RNNParameter(ProjectPath):
    def __init__(self,column,DIM_NUM):
        ProjectPath.__init__(self)
        self.num_outputs = 19
        self.lr = 0.1
        self.num_epochs = 1
        self.batch_size = 256
        self.num_hiddens = 256
        self.num_layers = 2
        self.bidirectional = True
        self.vocab_file = os.path.join(self.data_dir,"{}.dict".format(column))
        self.embedding_file = os.path.join(self.data_dir,"{}.{}d.txt".format(column,DIM_NUM))
        self.best_param_file = os.path.join(self.model_dir,"rnn_{}_best.param".format(column))

class MLParameter(ProjectPath):
    def __init__(self,column,lsi_num,lda_num):
        ProjectPath.__init__(self)
        lsi_file = os.path.join(self.model_dir,"lsi_{}_{}d.model".format(column,lsi_num))
        tfidf_ifle = os.path.join(self.model_dir,"tfidf_{}.model".format(column))
        lda_file = os.path.join(self.model_dir,"lda_{}_{}d.model".format(column,lda_num))




