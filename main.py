import torch
from collections import Counter
from models import EncoderDecoder
from data import TextDataset, PadAndOneHot

def train(model, dataset):
	# shuffle indices

	# 
	return 1

def test(model, dataset):
	return 1

# To use a different training text file, just change this path.
# Each line separated by '\n' will be used as one training example.
with open("war_and_peace.txt", "r") as f:
	data = f.readlines()
data[-1] += '\n'

# get size of input and output alphabets
c = Counter(("".join(data)))
Sy = list(c.keys()) # set of possible output letters
Sy_size = len(Sy) # 82, including EOS
Sx = [letter for letter in Sy if letter not in "AEIOUaeiou"] # remove vowels from set of possible input letters
Sx_size = len(Sx) # 72, including EOS
EOS_token = '\n' # all sequences end with newline
x_eos = Sx.index(EOS_token)
y_eos = Sy.index(EOS_token)
collate_fn = PadAndOneHot(Sx, Sy, x_eos, y_eos) # function for generating a minibatch from strings

# split dataset
total_lines = len(data)
one_tenth = total_lines // 10

train_dataset = TextDataset(data[0:one_tenth * 8])
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True, collate_fn=collate_fn)

valid_dataset = TextDataset(data[one_tenth * 8: one_tenth * 9])
test_dataset = TextDataset(data[one_tenth * 9:])

# initialize model
model = EncoderDecoder(	num_encoder_layers=2,
						num_encoder_hidden=128, 
						num_decoder_layers=1, 
						num_decoder_hidden=128, 
						Sx_size=len(Sx), 
						Sy_size=len(Sy),
						y_eos=y_eos,
						dropout=0.5)
if torch.cuda.is_available(): model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
	for idx, batch in enumerate(train_data_loader):
		x,y = batch
		if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
		log_probs = model(x,y); U = y.shape[1]
		loss = -log_probs.mean() / U
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss)
		if idx % 20 == 0:
			model.eval()
			y_hat = model.infer(x, Sy)
			model.train()
			print("".join([Sy[c] for c in y[0].max(dim=1)[1] if c != 25]))
			print("".join([Sy[c] for c in y_hat[0].max(dim=1)[1] if c != 25]))

test_output = "Hello, world!"
test_input = "".join([c for c in test_output if c not in "AEIOUaeiou"])
# print(model.infer(test_input))
