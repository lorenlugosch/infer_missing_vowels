import torch
from collections import Counter
from models import EncoderDecoder
from helper_functions import one_hot

# To use a different training text file, just change this path.
# Each line separated by '\n' will be used as one training example.
with open("war_and_peace.txt", "r") as f:
	lines = f.readlines()

c = Counter(("".join(lines)))
Sy = list(c.keys()) # set of possible output letters
Sy_size = len(Sy) # 82, including EOS
Sx = [letter for letter in Sy if letter not in "AEIOUaeiou"] # remove vowels from set of possible input letters
Sx_size = len(Sx) # 72, including EOS
EOS_token = '\n' # all sequences end with newline
y_eos = Sy.index(EOS_token)

model = EncoderDecoder(	num_encoder_layers=2,
						num_encoder_hidden=128, 
						num_decoder_layers=2, 
						num_decoder_hidden=128, 
						Sx_size=Sx_size, 
						Sy_size=Sy_size,
						y_eos=y_eos)


