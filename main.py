import torch
from models import EncoderDecoder
from data import get_datasets, PadAndOneHot
from training import Trainer

path = "war_and_peace.txt"
train_dataset, valid_dataset, test_dataset = get_datasets(path)

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

trainer = Trainer(model, lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
	print("========= Epoch %d of %d =========" % (epoch, num_epochs))
	train_acc, train_loss = trainer.train(train_dataset)
	valid_acc, valid_loss = trainer.test(valid_dataset)

	print("========= Results: epoch %d of %d =========" % (epoch, num_epochs))
	print("train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_acc, train_loss, valid_acc, valid_loss) )

	torch.save(model, "model.pth")

#############################
# test_output = "Hello, world!\n"
# test_input = 'Hll, wrld!\n' #"".join([c for c in test_output if c not in "AEIOUaeiou"])
# x,y = pad_and_one_hot([(test_input, test_output)])
# if torch.cuda.is_available(): x = x.cuda()
# y_hat = model.infer(x, Sy)
# print("input: " + "".join([Sx[c] for c in x[0].max(dim=1)[1] if c != x_eos]))
# print("truth: " + "".join([Sy[c] for c in y[0].max(dim=1)[1] if c != y_eos]))
# print("guess: " + "".join([Sy[c] for c in y_hat[0].max(dim=1)[1] if c != y_eos]))
# print("")
# #############################