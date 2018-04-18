import fire
import torch
from torch.autograd import Variable


class RNN(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.lin_hidden = torch.nn.Linear(in_size + hidden_size, hidden_size)
        self.lin_out = torch.nn.Linear(in_size + hidden_size, out_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden))
        hidden = self.lin_hidden(combined).clamp(min=0)
        out = self.lin_out(combined)
        return out, hidden


def main():
    in_size = 1
    hidden_size = 3
    out_size = 1
    rnn = RNN(in_size, hidden_size, out_size)

    x = Variable(torch.FloatTensor(torch.randn(in_size)))
    hidden = Variable(torch.FloatTensor(torch.randn(hidden_size)))

    print("x =", x)
    print("hidden =", hidden)

    for i in range(10):
        x, hidden = rnn(x, hidden)
        print("x =", x)
        print("hidden =", hidden)


def learn_sine():
    in_size = 1
    hidden_size = 100
    out_size = 1

    rnn = RNN(in_size, hidden_size, out_size)


fire.Fire()