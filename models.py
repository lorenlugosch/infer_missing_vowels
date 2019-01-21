import torch
from helper_functions import one_hot

class EncoderDecoder(torch.nn.Module):
	def __init__(self, num_encoder_layers, num_encoder_hidden, num_decoder_layers, num_decoder_hidden, Sx_size, Sy_size, y_eos, dropout):
		super(EncoderDecoder, self).__init__()
		self.encoder_rnn = torch.nn.GRU(input_size=Sx_size, hidden_size=num_encoder_hidden, num_layers=num_encoder_layers, batch_first=True, dropout=dropout, bidirectional=True) #EncoderRNN(num_encoder_layers, num_encoder_hidden)
		self.encoder_linear = torch.nn.Linear(num_encoder_hidden*2, num_decoder_hidden)
		self.decoder_rnn = torch.nn.GRUCell(input_size=Sy_size, hidden_size=num_decoder_hidden) #DecoderRNN(num_decoder_layers, num_decoder_hidden)
		self.decoder_linear = torch.nn.Linear(num_decoder_hidden, Sy_size)
		self.decoder_log_softmax = torch.nn.LogSoftmax(dim=1)
		self.y_eos = y_eos # index of the end-of-sequence token

	def forward(x, y):
		"""
		x : Tensor of shape (batch size, |Sx|, T)
		y : Tensor of shape (batch size, |Sy|, U) - padded with end-of-sequence tokens

		Compute log p(y|x) for each (x,y) in the batch.
		"""

		batch_size = y.shape[0]
		Sy_size = y.shape[1]
		U = y.shape[2]

		# Encode the input sequence into a single fixed-length vector
		_, encoder_state = self.encoder_rnn(x)
		encoder_state = torch.cat([encoder_state[-1], encoder_state[-2]])
		print(x.shape)
		print(encoder_state.shape)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)

		# Initialize log p(y|x) to zeros
		log_p_y_x = torch.zeros(batch_size)
		for u in range(0, U):
			if u == 0:
				# Feed in 0
				decoder_state = self.decoder_rnn(torch.zeros(batch_size, Sy_size), decoder_state)
			else:
				# Feed in the previous element of y; update the decoder state
				decoder_state = self.decoder_rnn(y[:,:,u-1], decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			out = self.decoder_linear(decoder_state)
			out = self.decoder_log_softmax(out)
			log_p_yu = (out * y[:,:,u]).sum() # y_u is one-hot; use dot-product to select the y_u'th output probability

			# Add log p(y_u|...) to log p(y|x)
			log_p_y_x += log_p_yu

		return log_p_y_x

	def infer(x, B=1):
		"""
		x : Tensor of shape (batch size, |Sx|, T)

		Run beam search to find y_hat = argmax_y log p(y|x) for every (x) in the batch.
		(If B = 1, this is equivalent to greedy search.) 
		"""
		y_hat = [] # List of guessed outputs for each input in the batch.

		U_max = 100 # maximum length
		for x_ in x:
			done = False
			beam = []
			while u < U_max:
				1

			y_hat.append(beam[0])

		return y_hat

