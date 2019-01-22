import torch
from helper_functions import one_hot
import sys

class EncoderRNN(torch.nn.Module):
	def __init__(self, num_encoder_layers, num_encoder_hidden, input_size, dropout):
		super(EncoderRNN, self).__init__()
		self.gru = torch.nn.GRU(input_size=input_size, hidden_size=num_encoder_hidden, num_layers=num_encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)

	def forward(self, input):
		"""
		input: Tensor of shape (batch size, T, |Sx|)
		
		Map the input sequence to a fixed-length encoding.
		"""
		_, out = self.gru(input)
		out = torch.cat([out[-1], out[-2]], dim=1)
		return out

class DecoderRNN(torch.nn.Module):
	def __init__(self, num_decoder_layers, num_decoder_hidden, input_size, dropout):
		super(DecoderRNN, self).__init__()
		self.gru = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden) 
		# self.layers = []
		# for index in range(num_decoder_layers):
		# 	if index == 0: 
		# 		layer = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden) 
		# 	else:
		# 		layer = torch.nn.GRUCell(input_size=num_decoder_hidden, hidden_size=num_decoder_hidden) 
		# 	self.layers.append(layer)

	def forward(self, input, previous_state):
		"""
		input: Tensor of shape (batch size, input_size)
		previous_state: Tensor of shape (batch size, num_decoder_hidden*num_decoder_layers)
		
		Given the input vector, update the hidden state of each decoder layer.
		"""
		return self.gru(input, previous_state)
		# state = []
		# batch_size = input.shape[0]
		# previous_state = previous_state.view(len(layers), batch_size, -1)
		# for index, layer in enumerate(self.layers):
		# 	if index = 0:
		# 		layer_out = layer(input, previous_state[index])
		# 		state.append(layer_out)
		# 	else:
		# 		layer_out = layer(layer_out, previous_state[index])
		# 		state.append(layer_out)
		# state = torch.stack(state).view(batch_size, -1)
		# return 

class EncoderDecoder(torch.nn.Module):
	"""
	Simple encoder-decoder sequence model with fixed-length encoding. 
	- forward(): computes the probability of an input/output sequence pair.
	- infer(): given the input sequence, infer the most likely output sequence.
	"""
	def __init__(self, num_encoder_layers, num_encoder_hidden, num_decoder_layers, num_decoder_hidden, Sx_size, Sy_size, y_eos, dropout):
		super(EncoderDecoder, self).__init__()
		self.encoder_rnn = EncoderRNN(num_encoder_layers, num_encoder_hidden, Sx_size, dropout)
		self.encoder_linear = torch.nn.Linear(num_encoder_hidden*2, num_decoder_hidden*num_decoder_layers)
		self.decoder_rnn = DecoderRNN(num_decoder_layers, num_decoder_hidden, Sy_size, dropout)
		self.decoder_linear = torch.nn.Linear(num_decoder_hidden, Sy_size)
		self.decoder_log_softmax = torch.nn.LogSoftmax(dim=1)
		self.y_eos = y_eos # index of the end-of-sequence token

	def forward(self, x, y):
		"""
		x : Tensor of shape (batch size, T, |Sx|)
		y : Tensor of shape (batch size, U, |Sy|) - padded with end-of-sequence tokens

		Compute log p(y|x) for each (x,y) in the batch.
		"""

		batch_size = y.shape[0]
		U = y.shape[1]
		Sy_size = y.shape[2]

		# Encode the input sequence into a single fixed-length vector
		encoder_state = self.encoder_rnn(x)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)

		# Initialize log p(y|x), y_u-1 to zeros
		log_p_y_x = 0
		y_u_1 = torch.zeros(batch_size, Sy_size)
		if torch.cuda.is_available(): y_u_1 = y_u_1.cuda()
		for u in range(0, U):
			# Feed in the previous element of y; update the decoder state
			decoder_state = self.decoder_rnn(y_u_1, decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state))
			log_p_yu = (decoder_out * y[:,u,:]).sum(dim=1) # y_u is one-hot; use dot-product to select the y_u'th output probability 

			# Add log p(y_u|...) to log p(y|x)
			log_p_y_x += log_p_yu

			# Look at next element of y
			y_u_1 = y[:,u,:]

		return log_p_y_x

	def infer(self, x, Sy, B=1):
		"""
		x : Tensor of shape (batch size, T, |Sx|)
		Sy : list of characters (output alphabet)

		Run beam search to find y_hat = argmax_y log p(y|x) for every (x) in the batch.
		(If B = 1, this is equivalent to greedy search.) 
		TODO implement B > 1
		"""
		# y_hat = [] # List of guessed outputs for each input in the batch.

		# U_max = 100 # maximum length
		# for x_ in x:
		# 	done = False
		# 	beam = []
		# 	while u < U_max:
		# 		break

		# 	y_hat.append(beam[0])

		# return y_hat

		batch_size = x.shape[0]
		Sy_size = len(Sy)

		# Encode the input sequence into a single fixed-length vector
		encoder_state = self.encoder_rnn(x)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)

		# Initialize list to empty
		y_hat = []
		U_max = 100
		y_hat_u_1 = torch.zeros(batch_size, Sy_size)
		if torch.cuda.is_available(): y_hat_u_1 = y_hat_u_1.cuda()
		for u in range(U_max):
			# Feed in the previous guess; update the decoder state
			# decoder_state = self.decoder_rnn(y_hat[:,u-1,:], decoder_state)
			decoder_state = self.decoder_rnn(y_hat_u_1, decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state))

			# Find the top output
			y_hat_u_1 -= y_hat_u_1 # set to zero
			y_hat_u_1[torch.arange(batch_size), decoder_out.max(dim=1)[1]] = 1.
			y_hat.append(y_hat_u_1.clone())

		return y_hat

