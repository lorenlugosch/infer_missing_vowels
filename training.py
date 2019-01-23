import torch

class Trainer:
	def __init__(self, model, lr):
		self.model = model
		self.lr = lr
		
	def train(self, dataset):
		self.model.train()
		train_acc = 0.
		train_loss = 0.
		return train_acc, train_loss

	def test(self, dataset):
		self.model.eval()
		test_acc = 0.
		test_loss = 0.
		return test_acc, test_loss