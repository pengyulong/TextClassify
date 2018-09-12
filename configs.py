import os

class ProjectPath:
	def __init__(self):
		self.data_dir = "../data"
		self.model_dir = "../model"
		self.train_file = os.path.join(self.data_dir,"train_set.csv")
		self.test_file = os.path.join(self.data_dir,"test_set.csv")

