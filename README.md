# infer_missing_vowels
This repository contains code for training a simple model to infer the original sentence when vowels are removed.

To test the model on a new input string, change `test_output` in `main.py` to the string you want, then run `python main.py`. (The next line of code removes the vowels to make `test_input`; you can also specify `test_input` manually.) The line `y_hat = model.infer(...)` runs a beam search to find the output and prints intermediate results; to turn off print statements, set `debug=False` in the function call.

To train the model, set `num_epochs` in `main.py` to the number of epochs you would like to train. (The model with the default hyperparameters written in `main.py` can be trained in 15 epochs.)

To change the dataset used for training, change the path given to `get_datasets()`. 
