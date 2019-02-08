import torch
from models import EncoderDecoder
from data import get_datasets, PadAndOneHot
from training import Trainer
from helper_functions import one_hot_to_string

# Generate datasets from text file
# (To use a difference text file, just change the path. The text file must contain strings separated by newlines.)
data_path = "war_and_peace.txt"
train_dataset, valid_dataset, test_dataset = get_datasets(data_path)
checkpoint_path = "."

# Initialize model
model = EncoderDecoder(	num_encoder_layers=2,
						num_encoder_hidden=512, 
						num_decoder_layers=2, 
						num_decoder_hidden=512,
						Sx_size=len(train_dataset.Sx),	# input alphabet
						Sy_size=len(train_dataset.Sy),	# output alphabet
						y_eos=train_dataset.y_eos,		# index of end-of-sequence symbol for output
						dropout=0.5)
if torch.cuda.is_available(): model = model.cuda()

# Train the model
num_epochs = 50
trainer = Trainer(model, lr=0.001)
trainer.load_checkpoint(checkpoint_path)

for epoch in range(num_epochs):
	print("========= Epoch %d of %d =========" % (epoch+1, num_epochs))
	train_acc, train_loss = trainer.train(train_dataset)
	valid_acc, valid_loss = trainer.test(valid_dataset)
	trainer.save_checkpoint(epoch, checkpoint_path)

	print("========= Results: epoch %d of %d =========" % (epoch+1, num_epochs))
	print("train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_acc, train_loss, valid_acc, valid_loss) )

# Example of testing the model on a new phrase
model.eval()
Sx = train_dataset.Sx; Sy = train_dataset.Sy; x_eos= train_dataset.x_eos; y_eos = train_dataset.y_eos
pad_and_one_hot = PadAndOneHot(Sx, Sy, x_eos, y_eos)

test_output = "It was the worst of times.\n"
test_output = test_output + "\n"*max(51-len(test_output),0) # pad so that length matches training distribution 
test_input = "".join([c for c in test_output if c not in "AEIOUaeiou"]) # 'Hll, wrld!\n'
x,y = pad_and_one_hot([(test_input, test_output)])
y_hat = model.infer(x, Sy, B=8, debug=True, true_U=len(test_output))
print("input: " + one_hot_to_string(x[0], Sx))
print("truth: " + one_hot_to_string(y[0], Sy))
print("guess: " + one_hot_to_string(y_hat[0], Sy))


