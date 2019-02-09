import torch
from helper_functions import one_hot, one_hot_to_string
import sys
import time

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

	def forward(self, input, x_lengths):
		"""
		input: Tensor of shape (batch size, T, |Sx|)
		x_lengths : list of integers
		
		Map the input sequence to a fixed-length encoding.
		"""
		# see https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
		# for how to use pad/pack
		sorted_lengths, sorting_indices = x_lengths.sort(0, descending=True)
		sorted_input = input[sorting_indices]
		packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths.cpu().numpy(), batch_first=True)
		sorted_outputs, sorted_final_state = self.gru(packed)
		sorted_outputs = torch.nn.utils.rnn.pad_packed_sequence(sorted_outputs)
		sorted_final_state = torch.cat([sorted_final_state[-1], sorted_final_state[-2]], dim=1)
		_, unsorting_indices = sorting_indices.sort(0)
		final_state = sorted_final_state[unsorting_indices]
		outputs = sorted_outputs[unsorting_indices]
		return outputs, final_state

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

def sort_beam(beam_extensions, beam_extension_scores, beam_pointers):
	beam_width = len(beam_pointers); batch_size = beam_pointers[0].shape[0]
	beam_extensions = torch.stack(beam_extensions); beam_extension_scores = torch.stack(beam_extension_scores); beam_pointers = torch.stack(beam_pointers)
	beam_extension_scores = beam_extension_scores.view(beam_width,batch_size)

	sort_order = beam_extension_scores.sort(dim=0, descending=True)[1].reshape(beam_width, batch_size)
	sorted_beam_extensions = beam_extensions.clone(); sorted_beam_extension_scores = beam_extension_scores.clone(); sorted_beam_pointers = beam_pointers.clone()
	
	for batch_index in range(batch_size):
		sorted_beam_extensions[:, batch_index] = beam_extensions[sort_order[:,batch_index], batch_index]
		sorted_beam_extension_scores[:, batch_index] = beam_extension_scores[sort_order[:,batch_index], batch_index]
		sorted_beam_pointers[:, batch_index] = beam_pointers[sort_order[:,batch_index], batch_index]
	return sorted_beam_extensions, sorted_beam_extension_scores, sorted_beam_pointers

class EncoderDecoder(torch.nn.Module):
	"""
	Simple encoder-decoder sequence model with fixed-length encoding. 
	- forward(): computes the probability of an input/output sequence pair.
	- infer(): given the input sequence, infer the most likely output sequence.
	"""
	def __init__(self, num_encoder_layers, num_encoder_hidden, num_decoder_layers, num_decoder_hidden, Sx_size, Sy_size, y_eos, dropout, use_attention):
		super(EncoderDecoder, self).__init__()
		self.encoder_rnn = EncoderRNN(num_encoder_layers, num_encoder_hidden, Sx_size, dropout)
		self.encoder_linear = torch.nn.Linear(num_encoder_hidden*2, num_decoder_hidden*num_decoder_layers)
		self.using_attention = use_attention
		key_dim = 100
		value_dim = 200
		if self.using_attention:
			self.decoder_init_state = torch.nn.Parameter(torch.randn(key_dim))
			self.attention = Attention(key_dim=key_dim, value_dim=value_dim)
			self.decoder_rnn = DecoderRNN(num_decoder_layers, num_decoder_hidden, Sy_size, dropout)
		else:
			self.decoder_rnn = DecoderRNN(num_decoder_layers, num_decoder_hidden, Sy_size + value_dim, dropout)
		self.decoder_linear = torch.nn.Linear(num_decoder_hidden, Sy_size)
		self.decoder_log_softmax = torch.nn.LogSoftmax(dim=1)
		self.y_eos = y_eos # index of the end-of-sequence token
		
	def forward(self, x, y, x_lengths=None, y_lengths=None):
		"""
		x : Tensor of shape (batch size, T, |Sx|)
		y : Tensor of shape (batch size, U, |Sy|) - padded with end-of-sequence tokens
		x_lengths : list of integers
		y_lengths : list of integers

		Compute log p(y|x) for each (x,y) in the batch.
		"""
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()

		batch_size = y.shape[0]
		U = y.shape[1]
		Sy_size = y.shape[2]

		# Encode the input sequence
		encoder_outputs, encoder_final_state = self.encoder_rnn(x, x_lengths)

		# Initialize the decoder state using the encoder state
		if self.using_attention:
			decoder_state = self.decoder_init_state
		else:
			decoder_state = self.encoder_linear(encoder_final_state)
			decoder_state = decoder_state.view(batch_size, self.decoder_rnn.num_layers, -1)

		# Initialize log p(y|x), y_u-1 to zeros
		log_p_y_x = 0
		y_u_1 = torch.zeros(batch_size, Sy_size)
		if torch.cuda.is_available(): y_u_1 = y_u_1.cuda()
		for u in range(0, U):
			# Feed in the previous element of y and the attention output; update the decoder state
			if self.using_attention:
				context = self.attention(encoder_outputs, decoder_state)
				decoder_input = torch.cat([y_u_1, context], dim=1)
			else:
				decoder_input = y_u_1
			decoder_state = self.decoder_rnn(decoder_input, decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state[:,-1]))
			log_p_yu = (decoder_out * y[:,u,:]).sum(dim=1) # y_u is one-hot; use dot-product to select the y_u'th output probability 

			# Add log p(y_u|...) to log p(y|x)
			log_p_y_x += log_p_yu

			# Look at next element of y
			y_u_1 = y[:,u,:]

		return log_p_y_x

	def infer(self, x, x_lengths, y_lengths, Sy, B=2, debug=False, true_U=None):
		"""
		x : Tensor of shape (batch size, T, |Sx|)
		Sy : list of characters (output alphabet)

		Run beam search to find y_hat = argmax_y log p(y|x) for every (x) in the batch.
		(If B = 1, this is equivalent to greedy search.)
		"""
		if torch.cuda.is_available(): x = x.cuda()

		batch_size = x.shape[0]
		Sy_size = len(Sy)

		# Encode the input sequence into a single fixed-length vector
		encoder_state = self.encoder_rnn(x, x_lengths)

		# Initialize the decoder state using the encoder state
		decoder_state = self.encoder_linear(encoder_state)
		decoder_state = decoder_state.view(batch_size, self.decoder_rnn.num_layers, -1)

		U_max = 100
		if true_U is None:
			true_U = U_max
		else:
			U_max = true_U

		decoder_state_shape = decoder_state.shape
		beam = torch.zeros(B,batch_size,U_max,Sy_size); beam_scores = torch.zeros(B,batch_size); decoder_states = torch.zeros(B,decoder_state_shape[0], decoder_state_shape[1], decoder_state_shape[2])
		if torch.cuda.is_available():
			beam = beam.cuda()
			beam_scores = beam_scores.cuda()
			decoder_states = decoder_states.cuda()

		for u in range(U_max):
			beam_extensions = []; beam_extension_scores = []; beam_pointers = []

			# Add a delay so that it's easier to read the outputs during debugging
			if debug and u < true_U:
				time.sleep(1)
				print("")

			for b in range(B):
				# Get previous guess
				if u == 0: 
					beam_score = beam_scores[b]
					y_hat_u_1 = torch.zeros(batch_size, Sy_size)
					if torch.cuda.is_available():
						beam_score = beam_score.cuda()
						y_hat_u_1 = y_hat_u_1.cuda()
				else: 
					# Select hypothesis (and corresponding decoder state/score) from beam
					y_hat = beam[b]
					decoder_state = decoder_states[b]
					beam_score = beam_scores[b]
					y_hat_u_1 = y_hat[:,u-1,:]

					# If in debug mode, print out the current beam
					if debug and u < true_U: print(one_hot_to_string(y_hat[0,:u], Sy).strip("\n") + " | score: %1.2f" % beam_score[0].item())

				# Feed in the previous guess; update the decoder state
				decoder_state = self.decoder_rnn(y_hat_u_1, decoder_state)
				decoder_states[b] = decoder_state.clone()

				# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
				decoder_out = self.decoder_log_softmax(self.decoder_linear(decoder_state[:,-1]))

				# Find the top B possible extensions for each of the B hypotheses
				top_B_extension_scores, top_B_extensions = decoder_out.topk(B)
				top_B_extension_scores = top_B_extension_scores.transpose(0,1); top_B_extensions = top_B_extensions.transpose(0,1)
				for extension_index in range(B):
					extension = torch.zeros(batch_size, Sy_size)
					extension_score = top_B_extension_scores[extension_index] + beam_score
					extension[torch.arange(batch_size), top_B_extensions[extension_index]] = 1.
					beam_extensions.append(extension.clone())
					beam_extension_scores.append(extension_score.clone())
					beam_pointers.append(torch.ones(batch_size).long() * b) # we need to remember which hypothesis this extension belongs to

				# At the first decoding timestep, there are no other hypotheses to extend.
				if u == 0: break

			# Sort the B^2 extensions
			beam_extensions, beam_extension_scores, beam_pointers = sort_beam(beam_extensions, beam_extension_scores, beam_pointers)
			old_beam = beam.clone(); old_beam_scores = beam_scores.clone(); old_decoder_states = decoder_states.clone()
			beam *= 0; beam_scores *= 0; decoder_states *= 0;

			# Pick the top B extended hypotheses
			for b in range(len(beam_extensions[:B])):
				for batch_index in range(batch_size):
					beam[b,batch_index] = old_beam[beam_pointers[b, batch_index],batch_index]
					beam[b,batch_index,u,:] = beam_extensions[b, batch_index] # append the extensions to each hypothesis
					beam_scores[b, batch_index] = beam_extension_scores[b, batch_index] # update the beam scores
					decoder_states[b, batch_index] = old_decoder_states[beam_pointers[b, batch_index],batch_index]

		y_hat = beam[0]
		return y_hat

