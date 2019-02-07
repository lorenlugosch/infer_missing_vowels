import torch
from tqdm import tqdm # for displaying progress bar
from helper_functions import one_hot_to_string

class Trainer:
	def __init__(self, model, lr):
		self.model = model
		self.lr = lr
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
		
	def train(self, dataset):
		train_acc = 0
		train_loss = 0
		num_samples = 0
		self.model.train()
		for idx, batch in enumerate(tqdm(dataset.loader)):
			x,y = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,y); U = y.shape[1]
			loss = -log_probs.mean() / U
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			# self.model.eval(); y_hat = self.model.infer(x, dataset.Sy); self.model.train()
			train_loss += loss.cpu().data.numpy().item() * batch_size
			# train_acc += edit_distance(y,y_hat) * batch_size
		train_loss /= num_samples
		train_acc /= num_samples
		return train_acc, train_loss

	def test(self, dataset, print_interval=20):
		test_acc = 0
		test_loss = 0
		num_samples = 0
		self.model.eval()
		for idx, batch in enumerate(dataset.loader):
			x,y = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,y); U = y.shape[1]
			loss = -log_probs.mean() / U
			test_loss += loss.cpu().data.numpy().item() * batch_size
			# test_acc += edit_distance(y,y_hat) * batch_size
			if idx % print_interval == 0:
				print("input: " + one_hot_to_string(x[0], dataset.Sx))
				print("truth: " + one_hot_to_string(y[0], dataset.Sy))
				y_hat = self.model.infer(x, dataset.Sy, B=1)
				print("greedy guess: " + one_hot_to_string(y_hat[0], dataset.Sy))
				y_hat = self.model.infer(x, dataset.Sy, B=4)
				print("beam guess: " + one_hot_to_string(y_hat[0], dataset.Sy))
				print("")

				print("input: " + one_hot_to_string(x[1], dataset.Sx))
				print("truth: " + one_hot_to_string(y[1], dataset.Sy))
				y_hat = self.model.infer(x, dataset.Sy, B=1)
				print("greedy guess: " + one_hot_to_string(y_hat[1], dataset.Sy))
				y_hat = self.model.infer(x, dataset.Sy, B=4)
				print("beam guess: " + one_hot_to_string(y_hat[1], dataset.Sy))
				print("")
		test_loss /= num_samples
		test_acc /= num_samples
		return test_acc, test_loss
		