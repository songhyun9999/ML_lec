import torch
import torch.nn as nn
import numpy as np

string = 'hello pytorch. how long can a rnn cell remember? show me your limit!'
print(len(string))
chars = 'abcdefghijklmnopqrstuvwxyz ?!.,:;01'

char_list = [i for i in chars]
n_letter = len(char_list)
print(n_letter)
print(char_list)

n_hidden = 35
learning_rate = 0.01
total_epochs = 1000

def stringToOneht(string):
    start = np.zeros(n_letter, dtype=int)
    end = np.zeros(n_letter, dtype=int)
    start[-2] = 1  # [0,0,0,...,0,1,0]
    end[-1] = 1    # [0,0,0,...,0,0,1]

    for i in string:
        idx = char_list.index(i)
        odata = np.zeros(n_letter, dtype=int)
        odata[idx] = 1
        start = np.vstack([start, odata])
    output = np.vstack([start, end])
    return output

print()
print(stringToOneht('test'))

def onehotToChar(onehot_d):
    onehot = torch.Tensor.numpy(onehot_d)
    return char_list[onehot.argmax()]

data = np.zeros(n_hidden, dtype=int)
data[5]=1
print(onehotToChar(torch.from_numpy(data)))

class RNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.ac_fn = nn.Tanh()

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.ac_fn(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

rnn = RNNet(n_letter, n_hidden, n_letter)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

one_hot = torch.from_numpy(stringToOneht(string)).type_as(torch.FloatTensor())
print()
print(one_hot)

for i in range(total_epochs):
    optimizer.zero_grad()
    hidden = rnn.init_hidden()
    total_loss = 0

    for j in range(one_hot.size()[0] - 1):
        input = one_hot[j:j+1, :]
        target = one_hot[j+1]
        hypothesis, hidden = rnn(input, hidden)
        loss = loss_func(hypothesis.view(-1), target.view(-1))
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    start = torch.zeros(1, n_letter)
    start[:, -2] = 1
    with torch.no_grad():
        hidden = rnn.init_hidden()
        input = start
        output_string = ''
        for i in range(len(string)):
            output, hidden = rnn(input, hidden)
            output_string += onehotToChar(output.data)
            input = output
    print('', output_string)





