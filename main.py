import torch
from collections import Counter
from models import EncoderDecoder
from helper_functions import one_hot

def train(model, dataset):
	return 1

def test(model, dataset):
	return 1

def get_batch(dataset, indices, Sx, Sy, x_eos, y_eos):
	x = []; y = []
	# convert letters to integers
	for index in indices:
		x.append([Sx.index(c) for c in dataset[0][index]])
		y.append([Sy.index(c) for c in dataset[1][index]])

	# get max sequence length
	T = max([len(x_) for x_ in x])
	U = max([len(y_) for y_ in y])

	# pad all sequences with EOS to have same length
	for index in range(len(x)):
		x[index] += [x_eos] * (T - len(x[index]))
		x[index] = torch.tensor(x[index])
		y[index] += [y_eos] * (U - len(y[index]))
		y[index] = torch.tensor(y[index])

	x = one_hot(torch.stack(x), len(Sx))
	y = one_hot(torch.stack(y), len(Sy))

	return (x,y)

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

total_lines = len(lines)

# split dataset
one_tenth = total_lines // 10
train_labels = lines[0:one_tenth * 8]
train_input = ["".join(c for c in line if c not in "AEIOUaeiou") for line in train_labels]
train_dataset = (train_input, train_labels)

valid_labels = lines[one_tenth * 8: one_tenth * 9]
valid_input = ["".join(c for c in line if c not in "AEIOUaeiou") for line in valid_labels]
valid_dataset = (valid_input, valid_labels)

test_labels = lines[one_tenth * 9:]
test_input = ["".join(c for c in line if c not in "AEIOUaeiou") for line in test_labels]
test_dataset = (test_input, test_labels)

# initialize model
model = EncoderDecoder(	num_encoder_layers=2,
						num_encoder_hidden=128, 
						num_decoder_layers=2, 
						num_decoder_hidden=128, 
						Sx_size=Sx_size, 
						Sy_size=Sy_size,
						y_eos=y_eos,
						dropout=0.5)

x,y = get_batch(train_dataset, [0,1], Sx, Sy, x_eos, y_eos)
model()
sys.exit()

num_epochs = 10
for epoch in range(num_epochs):
	train(model, train_dataset)
	test(model, valid_dataset)

test_input = "Hello, world!"
# print(model.infer(test_input))
