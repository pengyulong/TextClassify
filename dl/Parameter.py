import os

class ProjectPath(object):
    def __init__(self,column,DIM_NUM):
        self.data_dir = '../data'
        self.model_dir = '../model'
        self.result_dir = '../result'
        self.train_file = os.path.join(self.data_dir,"train_set.csv")
        self.test_file = os.path.join(self.data_dir,"test_set.csv")
        self.vocab_file = os.path.join(self.data_dir,"{}.dict".format(column))
        self.embedding_file = os.path.join(self.data_dir,"{}.{}d.txt".format(column,DIM_NUM))


class CNNParameter(ProjectPath):
    def __init__(self,column,DIM_NUM):
        ProjectPath.__init__(self,column,DIM_NUM)
        self.best_param_file = os.path.join(self.model_dir,"cnn_{}_best.param".format(column))
        self.num_outputs = 19
        self.num_epochs = 10
        self.lr = 0.01
        self.batch_size = 64
        self.ngram_kernel_sizes = [3,4,5]
        self.nums_channels = [100,100,100]
        

class RNNParameter(ProjectPath):
    def __init__(self,column,DIM_NUM):
        ProjectPath.__init__(self,column,DIM_NUM)
        self.best_param_file = os.path.join(self.model_dir,"rnn_{}_best.param".format(column))
        self.num_outputs = 19
        self.lr = 0.1
        self.num_epochs = 100
        self.batch_size = 256
        self.num_hiddens = 256
        self.num_layers = 2
        self.bidirectional = True


