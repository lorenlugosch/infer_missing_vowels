import torch
from collections import Counter
from models import EncoderDecoder
from helper_functions import one_hot

import torch.utils.data

class TextDataset(torch.utils.data.Dataset):
	def __init__(self, lines):
		self.lines = lines

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		x = "".join(c for c in lines[idx] if c not in "AEIOUaeiou")
		y = lines[idx]
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
			y.append([self.Sy.index(c) for c in x_])

		# pad all sequences with EOS to have same length
		T = max([len(x_) for x_ in x])
		U = max([len(y_) for y_ in y])
		for index in range(batch_size):
			x[index] += [self.x_eos] * (T - len(x[index]))
			x[index] = torch.tensor(x[index])
			y[index] += [self.y_eos] * (U - len(y[index]))
			y[index] = torch.tensor(y[index])

		# stack into single tensor and one-hot encode integer labels
		x = one_hot(torch.stack(x), len(Sx))
		y = one_hot(torch.stack(y), len(Sy))

		return (x,y)


def train(model, dataset):
	# shuffle indices

	# 
	return 1

def test(model, dataset):
	return 1

# To use a different training text file, just change this path.
# Each line separated by '\n' will be used as one training example.
with open("war_and_peace.txt", "r") as f:
	lines = f.readlines()
lines[-1] += '\n'

# get size of input and output alphabets
c = Counter(("".join(lines)))
Sy = list(c.keys()) # set of possible output letters
Sy_size = len(Sy) # 82, including EOS
Sx = [letter for letter in Sy if letter not in "AEIOUaeiou"] # remove vowels from set of possible input letters
Sx_size = len(Sx) # 72, including EOS
EOS_token = '\n' # all sequences end with newline
x_eos = Sx.index(EOS_token)
y_eos = Sy.index(EOS_token)
collate_fn = PadAndOneHot(Sx, Sy, x_eos, y_eos) # function for generating a minibatch from strings

# split dataset
total_lines = len(lines)
one_tenth = total_lines // 10

train_dataset = TextDataset(lines[0:one_tenth * 8])
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=1, shuffle=True)

valid_dataset = TextDataset(lines[one_tenth * 8: one_tenth * 9])
test_dataset = TextDataset(lines[one_tenth * 9:])

# initialize model
model = EncoderDecoder(	num_encoder_layers=2,
						num_encoder_hidden=128, 
						num_decoder_layers=1, 
						num_decoder_hidden=128, 
						Sx_size=len(Sx), 
						Sy_size=len(Sy),
						y_eos=y_eos,
						dropout=0.5)

x,y = get_batch(train_dataset, [0,1], Sx, Sy, x_eos, y_eos)
log_probs = model(x,y); U = x.shape[1]
loss = -log_probs.mean() / U
sys.exit()

num_epochs = 10
for epoch in range(num_epochs):
	train(model, train_dataset)
	test(model, valid_dataset)

test_input = "Hello, world!"
# print(model.infer(test_input))
