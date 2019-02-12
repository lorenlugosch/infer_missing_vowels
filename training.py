import torch
from tqdm import tqdm # for displaying progress bar
from helper_functions import one_hot_to_string
import os
import pandas as pd

class Trainer:
	def __init__(self, model, lr):
		self.model = model
		self.lr = lr
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
		self.train_df = pd.DataFrame(columns=["loss","lr"])
		self.valid_df = pd.DataFrame(columns=["loss","lr"])

	def load_checkpoint(self, checkpoint_path):
		if os.path.isfile(os.path.join(checkpoint_path, "model_state.pth")):
			try:
				self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model_state.pth")))
			except:
				print("Could not load previous model; starting from scratch")
		else:
			print("No previous model; starting from scratch")

	def save_checkpoint(self, epoch, checkpoint_path):
		try:
			torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "model_state.pth"))
		except:
			print("Could not save model")
		
	def train(self, dataset):
		train_acc = 0
		train_loss = 0
		num_samples = 0
		self.model.train()
		for idx, batch in enumerate(tqdm(dataset.loader)):
			x,y,x_lengths,y_lengths = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,y,x_lengths,y_lengths); U = y.shape[1]
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
			x,y,x_lengths,y_lengths = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,y,x_lengths,y_lengths); U = y.shape[1]
			loss = -log_probs.mean() / U
			test_loss += loss.cpu().data.numpy().item() * batch_size
			# test_acc += edit_distance(y,y_hat) * batch_size
			if idx % print_interval == 0:
				self.model.is_cuda = False # Beam search may cause a GPU out-of-memory---do this on the CPU, for now
				self.model.cpu()
				x = x[:2]; y = y[:2]; x_lengths = x_lengths[:2]; y_lengths = y_lengths[:2]
				y_hat_greedy = self.model.infer(x,x_lengths,y_lengths,dataset.Sy, true_U=U, B=1)
				y_hat_beam = self.model.infer(x,x_lengths,y_lengths,dataset.Sy, true_U=U, B=4)
				print("input: " + one_hot_to_string(x[0], dataset.Sx))
				print("truth: " + one_hot_to_string(y[0], dataset.Sy))
				print("greedy guess: " + one_hot_to_string(y_hat_greedy[0], dataset.Sy))
				print("beam guess: " + one_hot_to_string(y_hat_beam[0], dataset.Sy))
				print("")
				print("input: " + one_hot_to_string(x[1], dataset.Sx))
				print("truth: " + one_hot_to_string(y[1], dataset.Sy))
				print("greedy guess: " + one_hot_to_string(y_hat_greedy[1], dataset.Sy))
				print("beam guess: " + one_hot_to_string(y_hat_beam[1], dataset.Sy))
				print("")
				self.model.is_cuda = True
				self.model.cuda()
		test_loss /= num_samples
		test_acc /= num_samples
		self.scheduler.step(test_loss) # if the validation loss hasn't decreased, lower the learning rate
		return test_acc, test_loss
		