import torch
from helper_functions import one_hot

class EncoderDecoder(torch.nn.Module):
	def __init__(self, Sx_size, Sy_size, y_eos):
		super(EncoderDecoder, self).__init__()
		self.encoder_rnn = 1
		self.encoder_linear = 1
		self.decoder_rnn = 1
		self.decoder_linear = 1
		self.decoder_log_softmax = 1
		self.y_eos = y_eos # index of the end-of-sequence token
		self.Sx_size = Sx_size
		self.Sy_size = Sy_size

	def forward(x, y):
		"""
		x : Tensor of shape (batch size, |Sx|, T)
		y : Tensor of shape (batch size, |Sy|, U) - padded with self.y_eos

		Compute log p(y|x) for each (x,y) in the batch.
		"""

		batch_size = y.shape[0]
		U = y.shape[2]

		# Encode the input sequence into a single fixed-length vector
		encoder_state = self.encoder_rnn(x)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)

		# Initialize log p(y|x) to zeros
		log_p_y_x = torch.zeros(batch_size)
		for u in range(0, U):
			if u == 0:
				# Feed in 0
				decoder_state = self.decoder_rnn(y[u-1], decoder_state)
			else:
				# Feed in the previous element of y; update the decoder state
				decoder_state = self.decoder_rnn(y[u-1], decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			out = self.decoder_linear(decoder_state)
			out = self.decoder_log_softmax(out)
			log_p_yu = out[:,y[u]] 

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

		for x_ in x:
			done = False
			beam = []
			while not done:
				for b in range(B):
					1

			y_hat.append(beam[0])

		return y_hat

