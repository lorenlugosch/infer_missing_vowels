import torch

class Trainer:
	def __init__(self, model, lr):
		self.model = model
		self.lr = lr
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		
	def train(self, dataset):
		train_acc = 0
		train_loss = 0
		num_samples = 0
		self.model.train()
		for idx, batch in enumerate(dataset.loader):
			x,y = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,y); U = y.shape[1]
			loss = -log_probs.mean() / U
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			self.model.eval(); y_hat = self.model.infer(x, dataset.Sy); self.model.train()
			train_loss += loss.cpu().data.numpy().item() * batch_size
			# train_acc += edit_distance(y,y_hat) * batch_size

			# print out the model's guess, to see how well it's learning
			if idx % 20 == 0:
				print("input: " + "".join([dataset.Sx[c] for c in x[0].max(dim=1)[1] if c != dataset.x_eos]))
				print("truth: " + "".join([dataset.Sy[c] for c in y[0].max(dim=1)[1] if c != dataset.y_eos]))
				print("guess: " + "".join([dataset.Sy[c] for c in y_hat[0].max(dim=1)[1] if c != dataset.y_eos]))
				print("")
		train_loss /= num_samples
		train_acc /= num_samples
		return train_acc, train_loss

	def test(self, dataset):
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
			y_hat = self.model.infer(x, dataset.Sy)
			test_loss += loss.cpu().data.numpy().item() * batch_size
			# test_acc += edit_distance(y,y_hat) * batch_size
		test_loss /= num_samples
		test_acc /= num_samples
		return test_acc, test_loss