import fire
import numpy as np
import torch
from torch.autograd import Variable


class RNN(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.lin_hidden = torch.nn.Linear(in_size + hidden_size, hidden_size)
        self.lin_out = torch.nn.Linear(in_size + hidden_size, out_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 0)
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


def get_sine():
    start = np.random.uniform(-10, 10)
    width = np.random.uniform(0.1, 20)
    x = torch.FloatTensor(torch.linspace(start, start + width))
    y = torch.FloatTensor(torch.sin(x))
    return Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def learn_sine():
    in_size = 1
    hidden_size = 100
    out_size = 1

    rnn = RNN(in_size, hidden_size, out_size)
    hidden = Variable(torch.zeros(hidden_size))
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=1e-5)

    for i in range(100):
        print("Step", i, end="; ", flush=True)
        xs, ys = get_sine()
        loss = 0
        for x, y_target in zip(xs, ys):
            y_pred, hidden = rnn(x, hidden)
            loss += criterion(y_pred, y_target)
        print("Loss =", loss.data[0])
        optimizer.zero_grad()
        loss.backward(retain_graph=(i<99))
        optimizer.step()


def test_cat():
    x = Variable(torch.FloatTensor(torch.randn(1)))
    y = Variable(torch.FloatTensor(torch.randn(10)))

    print(torch.cat((x, y), 0))



fire.Fire()