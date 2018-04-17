import fire
import torch
from torch.autograd import Variable


def main():
    x = Variable(torch.randn(1).type(torch.FloatTensor), requires_grad=True)
    print("x =", x.data[0])
    y = x**2
    print("y = x**2")

    y.backward()
    print("dy/dx =", x.grad.data[0])


fire.Fire()