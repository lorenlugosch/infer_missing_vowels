import torch
from models import EncoderDecoder
from data import get_datasets, PadAndOneHot
from training import Trainer

# To use a different training text file, just change this path.
# Each line separated by '\n' will be used as one training example.
# with open("war_and_peace.txt", "r") as f:
# 	data = f.readlines()
# data[-1] += '\n'

# # split dataset
# total_lines = len(data)
# one_tenth = total_lines // 10

# train_dataset = TextDataset(data[0:one_tenth * 8])
# # train_data_loader = train_dataset.loader

# valid_dataset = TextDataset(data[one_tenth * 8: one_tenth * 9])
# # valid_data_loader = valid_dataset.loader

# test_dataset = TextDataset(data[one_tenth * 9:])

path = "war_and_peace.txt"
train_dataset, valid_dataset, test_dataset = get_datasets(path)
train_data_loader = train_dataset.loader
valid_data_loader = valid_dataset.loader

# initialize model
model = EncoderDecoder(	num_encoder_layers=2,
						num_encoder_hidden=128, 
						num_decoder_layers=1, 
						num_decoder_hidden=128,
						Sx_size=len(train_dataset.Sx),	# input alphabet
						Sy_size=len(train_dataset.Sy),	# output alphabet
						y_eos=train_dataset.y_eos,		# index of end-of-sequence symbol for output
						dropout=0.5)
if torch.cuda.is_available(): model = model.cuda()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = training.Trainer(model, lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
	print("========= Epoch %d of %d =========" % (epoch, num_epochs))

	# train
	train_acc = 0
	train_loss = 0
	num_samples = 0
	model.train()
	for idx, batch in enumerate(train_data_loader):
		x,y = batch
		batch_size = len(x)
		num_samples += batch_size
		if torch.cuda.is_available(): # move to inside of model.forward()?
				x = x.cuda()
				y = y.cuda()
		log_probs = model(x,y); U = y.shape[1]
		loss = -log_probs.mean() / U
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		model.eval(); y_hat = model.infer(x, Sy); model.train()
		train_loss += loss.cpu().data.numpy().item() * batch_size
		# train_acc += edit_distance(y,y_hat) * batch_size

		# print out the model's guess, to see how well it's learning
		if idx % 20 == 0:
			print("input: " + "".join([Sx[c] for c in x[0].max(dim=1)[1] if c != x_eos]))
			print("truth: " + "".join([Sy[c] for c in y[0].max(dim=1)[1] if c != y_eos]))
			print("guess: " + "".join([Sy[c] for c in y_hat[0].max(dim=1)[1] if c != y_eos]))
			print("")
	train_loss /= num_samples
	train_acc /= num_samples

	# valid
	valid_acc = 0
	valid_loss = 0
	num_samples = 0
	model.eval()
	for idx, batch in enumerate(valid_data_loader):
		x,y = batch
		batch_size = len(x)
		num_samples += batch_size
		if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
		log_probs = model(x,y); U = y.shape[1]
		loss = -log_probs.mean() / U
		y_hat = model.infer(x, Sy)
		valid_loss += loss.cpu().data.numpy().item() * batch_size
		# valid_acc += edit_distance(y,y_hat) * batch_size
	valid_loss /= num_samples
	valid_acc /= num_samples

	#############################
	test_output = "Hello, world!\n"
	test_input = 'Hll, wrld!\n' #"".join([c for c in test_output if c not in "AEIOUaeiou"])
	x,y = pad_and_one_hot([(test_input, test_output)])
	if torch.cuda.is_available(): x = x.cuda()
	y_hat = model.infer(x, Sy)
	print("input: " + "".join([Sx[c] for c in x[0].max(dim=1)[1] if c != x_eos]))
	print("truth: " + "".join([Sy[c] for c in y[0].max(dim=1)[1] if c != y_eos]))
	print("guess: " + "".join([Sy[c] for c in y_hat[0].max(dim=1)[1] if c != y_eos]))
	#############################

	print("========= Results: epoch %d of %d =========" % (epoch, num_epochs))
	print("train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f" % (train_acc, train_loss, valid_acc, valid_loss) )
	print("")

	torch.save(model, "model.pth")