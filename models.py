import torch
from helper_functions import one_hot
import sys

class Attention(torch.nn.Module):
	def __init__(self, key_dim, value_dim):
		super(Attention, self).__init__()
		self.scale_factor = torch.sqrt(torch.tensor(key_dim).float())
		self.key_linear = torch.nn.Linear(input_dim, key_dim)
		self.value_linear = torch.nn.Linear(input_dim, value_dim)
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, input, query):
		"""
		input: Tensor of shape (batch size, T, input_dim)
		query: Tensor of shape (batch size, key_dim)
		
		Map the input sequences to vectors (batch size, value_dim) using attention, given a query.
		"""
		keys = self.key_linear(input)
		values = self.value_linear(input)
		query = query.unsqueeze(2)
		scores = torch.matmul(keys, query) / self.scale_factor
		normalized_scores = self.softmax(scores).transpose(1,2)
		out = torch.matmul(normalized_scores, values).squeeze(1)
		return out

class EncoderRNN(torch.nn.Module):
	def __init__(self, num_encoder_layers, num_encoder_hidden, input_size, dropout):
		super(EncoderRNN, self).__init__()
		self.gru = torch.nn.GRU(input_size=input_size, hidden_size=num_encoder_hidden, num_layers=num_encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)

	def forward(self, input):
		"""
		input: Tensor of shape (batch size, T, |Sx|)
		
		Map the input sequence to a fixed-length encoding.
		"""
		_, final_state = self.gru(input)
		final_state = torch.cat([final_state[-1], final_state[-2]], dim=1)
		return final_state

class DecoderRNN(torch.nn.Module):
	def __init__(self, num_decoder_layers, num_decoder_hidden, input_size, dropout):
		super(DecoderRNN, self).__init__()
		# self.gru = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden)
		# self.dropout = torch.nn.Dropout(dropout)

		self.layers = []
		self.num_layers = num_decoder_layers
		for index in range(num_decoder_layers):
			if index == 0: 
				layer = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden) 
			else:
				layer = torch.nn.GRUCell(input_size=num_decoder_hidden, hidden_size=num_decoder_hidden) 
			layer.name = "gru%d"%index
			self.layers.append(layer)

			layer = torch.nn.Dropout(p=dropout)
			layer.name = "dropout%d"%index
			self.layers.append(layer)
		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, input, previous_state):
		"""
		input: Tensor of shape (batch size, input_size)
		previous_state: Tensor of shape (batch size, num_decoder_layers, num_decoder_hidden)
		
		Given the input vector, update the hidden state of each decoder layer.
		"""
		# return self.gru(input, previous_state)

		state = []
		batch_size = input.shape[0]
		gru_index = 0
		for index, layer in enumerate(self.layers):
			if index == 0:
				layer_out = layer(input, previous_state[:, gru_index])
				state.append(layer_out)
				gru_index += 1
			else:
				if "gru" in layer.name:
					layer_out = layer(layer_out, previous_state[:, gru_index])
					state.append(layer_out)
					gru_index += 1
				else: 
					layer_out = layer(layer_out)
		state = torch.stack(state, dim=1)
		return state 

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
		# self.attention = Attention(key_dim=100, value_dim=200)

	def forward(self, x, y):
		"""
		x : Tensor of shape (batch size, T, |Sx|)
		y : Tensor of shape (batch size, U, |Sy|) - padded with end-of-sequence tokens

		Compute log p(y|x) for each (x,y) in the batch.
		"""
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()

		batch_size = y.shape[0]
		U = y.shape[1]
		Sy_size = y.shape[2]

		# Encode the input sequence into a single fixed-length vector
		encoder_state = self.encoder_rnn(x)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)
		decoder_state = decoder_state.view(batch_size, self.decoder_rnn.num_layers, -1)

		# Initialize log p(y|x), y_u-1 to zeros
		log_p_y_x = 0
		y_u_1 = torch.zeros(batch_size, Sy_size)
		if torch.cuda.is_available(): y_u_1 = y_u_1.cuda()
		for u in range(0, U):
			# Feed in the previous element of y and the attention output; update the decoder state
			# context = self.attention(encoder_outputs)
			# decoder_input = torch.cat([y_u_1, context])
			# decoder_state = self.decoder_rnn(decoder_input, decoder_state)

			# Feed in the previous element of y; update the decoder state
			decoder_state = self.decoder_rnn(y_u_1, decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state[:,-1]))
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
		greedy = True

		if torch.cuda.is_available(): x = x.cuda()

		batch_size = x.shape[0]
		Sy_size = len(Sy)

		# Encode the input sequence into a single fixed-length vector
		encoder_state = self.encoder_rnn(x)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)
		decoder_state = decoder_state.view(batch_size, self.decoder_rnn.num_layers, -1)

		# Initialize list to empty
		y_hat = []
		U_max = 100

		if greedy:
			for u in range(U_max):
				# Get previous guess
				if u == 0: y_hat_u_1 = torch.zeros(batch_size, Sy_size)
				else: y_hat_u_1 = y_hat[-1]
				if torch.cuda.is_available(): y_hat_u_1 = y_hat_u_1.cuda()

				# Feed in the previous guess; update the decoder state
				decoder_state = self.decoder_rnn(y_hat_u_1, decoder_state)

				# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
				decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state[:,-1]))

				# Find the top output
				extension = torch.zeros(batch_size, Sy_size)
				extension[torch.arange(batch_size), decoder_out.max(dim=1)[1]] = 1.
				y_hat.append(extension.clone())

		# else: 
		# 	beam = []; beam_extensions = []; decoder_states = []; decoder_state_extensions = []; log_prob
		# 	for u in range(U_max):
		# 		for b in range(B):
		# 			y_hat_u_1 = beam[b]
		# 			decoder_state = decoder_states[b]

		# 			# Feed in the previous guess; update the decoder state
		# 			# decoder_state = self.decoder_rnn(y_hat[:,u-1,:], decoder_state)
		# 			decoder_state = self.decoder_rnn(y_hat_u_1, decoder_state)

		# 			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
		# 			decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state[:,-1]))

		# 			# Find the top B outputs
		# 			beam_extensions.append() = decoder_out.topk(B)[1]
		# 			y_hat_u_1 -= y_hat_u_1 # set to zero
		# 			y_hat_u_1[torch.arange(batch_size), decoder_out.max(dim=1)[1]] = 1.
		# 			y_hat.append(y_hat_u_1.clone())
		# 		beam = 

		y_hat = torch.cat([y_.unsqueeze(1) for y_ in y_hat], dim=1)
		return y_hat

