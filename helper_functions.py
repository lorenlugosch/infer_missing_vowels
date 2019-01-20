import torch

def one_hot(letters, S):
	"""
	letters : LongTensor of shape (batch size, sequence length)
	S : integer

	Convert batch of integer letter indices to one-hot vectors of dimension S (# of possible letters).
	"""
	out = torch.zeros(letters.shape[0], S, letters.shape[1]).long()
	for i in range(0, letters.shape[0]):
		for t in range(0, letters.shape[1]):
			out[i,letters[i,t], t] = 1
	return out
	