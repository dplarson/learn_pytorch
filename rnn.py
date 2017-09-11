'''
Use Recurrent Neural Networks to predict a time-series.
'''

import torch
from torch import nn

input_size = 10     # number of input sequences
hidden_size = 20    # size of each hidden layer
num_layers = 2      # number of hidden layers
num_directions = 1  # number of directions
batch_size = 3      # number of samples per batch
seq_len = 5         # length of each input sequence

# create RNN model
model = nn.RNN(input_size, hidden_size, num_layers)
print model

# create synthetic data
inputs = torch.autograd.Variable(torch.randn(seq_len, batch_size, input_size))
h0 = torch.autograd.Variable(torch.randn(num_layers * num_directions, batch_size, hidden_size))
print inputs.size(), h0.size()

output, hn = model(inputs, h0)

print output.size(), hn.size()
