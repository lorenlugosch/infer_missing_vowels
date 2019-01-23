import torch
import torch.utils.data
from collections import Counter
from helper_functions import one_hot

def get_datasets(path):
	with open(path, "r") as f:
		lines = f.readlines()
	lines[-1] += '\n'

	# get input and output alphabets
	Sy = list(Counter(("".join(lines))).keys()) # set of possible output letters
	Sx = [letter for letter in self.Sy if letter not in "AEIOUaeiou"] # remove vowels from set of possible input letters

	# split dataset
	total_lines = len(lines)
	one_tenth = total_lines // 10

	train_dataset = TextDataset(data[0:one_tenth * 8], Sx, Sy)
	valid_dataset = TextDataset(data[one_tenth * 8: one_tenth * 9], Sx, Sy)
	test_dataset = TextDataset(data[one_tenth * 9:], Sx, Sy)

	return train_dataset, valid_dataset, test_dataset

class TextDataset(torch.utils.data.Dataset):
	def __init__(self, lines, Sx, Sy):
		self.lines = lines
		self.Sx = Sx
		self.Sy = Sy
		EOS_token = '\n' # all sequences end with newline
		self.x_eos = self.Sx.index(EOS_token)
		self.y_eos = self.Sy.index(EOS_token)
		pad_and_one_hot = PadAndOneHot(self.Sx, self.Sy, self.x_eos, self.y_eos) # function for generating a minibatch from strings
		
		self.loader = torch.utils.data.DataLoader(self, batch_size=32, num_workers=1, shuffle=True, collate_fn=pad_and_one_hot)

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		x = "".join(c for c in self.lines[idx] if c not in "AEIOUaeiou") # remove vowels
		y = self.lines[idx]
		return (x,y)

class PadAndOneHot:
	def __init__(self, Sx, Sy, x_eos, y_eos):
		self.Sx = Sx
		self.Sy = Sy
		self.x_eos = x_eos
		self.y_eos = y_eos

	def __call__(self, batch):
		"""
		batch: list of tuples (input string, output string)

		Returns a minibatch of strings, one-hot encoded and padded to have the same length.
		"""
		x = []; y = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_ = batch[index]

			# convert letters to integers
			x.append([self.Sx.index(c) for c in x_])
			y.append([self.Sy.index(c) for c in y_])

		# pad all sequences with EOS to have same length
		T = max([len(x_) for x_ in x])
		U = max([len(y_) for y_ in y])
		for index in range(batch_size):
			x[index] += [self.x_eos] * (T - len(x[index]))
			x[index] = torch.tensor(x[index])
			y[index] += [self.y_eos] * (U - len(y[index]))
			y[index] = torch.tensor(y[index])

		# stack into single tensor and one-hot encode integer labels
		x = one_hot(torch.stack(x), len(self.Sx))
		y = one_hot(torch.stack(y), len(self.Sy))

		return (x,y)
