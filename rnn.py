'''
Use Recurrent Neural Networks to predict a time-series.
'''

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create synthetic data for testing
# - target: sin(t)
# - input: corrupted sin(t) signals = sin(t) + noise
# - organization: match pandas DataFrame
#
t = np.linspace(0, 20 * np.pi, 1000)
y = np.sin(t)
x1 = y + (np.random.rand(y.shape[0]) - 0.5)
x2 = y + (np.random.rand(y.shape[0]) - 0.5)
x3 = y + (np.random.rand(y.shape[0]) - 0.5)

data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})

# created features matrix (lagged values)
feature_cols = ['x1', 'x2', 'x3']
features = pd.concat([
    data[feature_cols].shift(0),
    data[feature_cols].shift(1),
    data[feature_cols].shift(2),
    data[feature_cols].shift(3),
    data[feature_cols].shift(4),
], axis=1)

df = pd.merge(data[['y']], features, how='inner', left_index=True, right_index=True)

# drop rows with missing data
df = df.dropna(how='any')

# select variables
inputs = df[['x1', 'x2', 'x3']].values
target = df['y'].values

hidden_size = 20  # size of each hidden layer (including output hidden layer)
num_layers = 2    # number of hidden layers

# number of input sequences
input_size = len(feature_cols)

# number of output sequences
output_size = 1

# length of each input sequence
seq_len = df['x1'].shape[1]

# number of samples per batch
batch_size = df.shape[0]  # number of samples per batch

#print "Input size:", input_size
#print "Sequence length:", seq_len
#print "Hidden layer size:", hidden_size

#==================================================
# re-shape inputs to required shape
# - before: (batch_size, seq_len * input_size)
# - after: (seq_len, batch_size, input_size)
inputs = inputs.reshape(batch_size, seq_len, input_size)
#inputs = np.swapaxes(inputs, 0, 1)

# convert to torch data types
inputs = torch.autograd.Variable(torch.Tensor(inputs).float()).cuda()
target = torch.autograd.Variable(torch.Tensor(target).float()).cuda()


# RNN model (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        # Forward propogate RNN
        #output, (hn, cn) = self.lstm(x, (h0, c0))
        output, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        output = self.fc(output[:, -1, :])
        return output


# Initialize model
rnn = RNN(input_size, hidden_size, num_layers, output_size)
rnn.cuda()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=1e-3)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = rnn(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    print epoch, loss.data[0]

# final output
y_true = target.data.cpu().numpy()
y_pred = outputs.data.cpu().numpy().flatten()

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

plt.plot(y_true, ls='-', c='0.5')
plt.plot(y_pred, ls='--', c='k')
plt.title('RMSE = {:.4f}'.format(rmse))
plt.show()
