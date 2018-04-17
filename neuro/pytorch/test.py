import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


class FeedForward(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.W1 = Variable(torch.randn(in_size, hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.b1 = Variable(torch.randn(hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.W2 = Variable(torch.randn(hidden_size, out_size).type(torch.FloatTensor), requires_grad=True)
        self.b2 = Variable(torch.randn(out_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, x):
        x2 = torch.tanh(x @ self.W1 + self.b1)
        return x2 @ self.W2 + self.b2

    def update(self, learning_rate):
        self.W1.data -= learning_rate * self.W1.grad.data
        self.b1.data -= learning_rate * self.b1.grad.data
        self.W2.data -= learning_rate * self.W2.grad.data
        self.b2.data -= learning_rate * self.b2.grad.data

    def set_grad_zero(self):
        self.W1.grad.data.zero_()
        self.b1.grad.data.zero_()
        self.W2.grad.data.zero_()
        self.b2.grad.data.zero_()


class FeedForwardSuperior(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # torch.nn.Linear already contains bias
        self.lin1 = torch.nn.Linear(in_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        hidden = torch.tanh(self.lin1(x))
        return self.lin2(hidden)


def main():
    x = Variable(torch.randn(1).type(torch.FloatTensor), requires_grad=True)
    print("x =", x.data[0])
    y = x**2
    print("y = x**2")

    y.backward()
    print("dy/dx =", x.grad.data[0])


def get_data():
    x = np.linspace(0, 2 * np.pi)
    y = np.sin(x)
    x_t = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=False)
    y_t = Variable(torch.from_numpy(y).type(torch.FloatTensor), requires_grad=False)
    return x_t, y_t

def ff_test():
    x_t, y_t = get_data()

    ff = FeedForward(50, 20, 50)
    learning_rate = 1e-3

    for i in range(200):
        y_pred = ff.forward(x_t)
        loss = ((y_pred - y_t)**2).sum()
        print("Loss:", loss.data[0])

        loss.backward()

        ff.update(learning_rate)
        ff.set_grad_zero()


    plt.plot(x_t.data.numpy(), y_pred.data.numpy())
    plt.show()


def ff_superior():
    x_t, y_t = get_data()

    ff = FeedForwardSuperior(50, 20, 50)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(ff.parameters(), lr=1e-3)

    for i in range(200):
        y_pred = ff(x_t)

        loss = criterion(y_pred, y_t)
        print("Loss =", loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(x_t.data.numpy(), y_pred.data.numpy())
    plt.show()


fire.Fire()