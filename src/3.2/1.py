import numpy as np

import torch


def derivative_numpy(f, x0, h=1e-6):
    x1 = x0 + h
    x2 = x0 - h
    return (f(x1) - f(x2)) / (2*h)

x_vals = np.array([0.5, 1.0, 1.23, 2.0])
print('numpy diff:', derivative_numpy(np.sin, x_vals))


def derivative_pytorch(f, x0, h=1e-6):
    x1 = x0 + h
    x2 = x0 - h
    return (f(x1) - f(x2)) / (2*h)

x_vals = torch.tensor([0.5, 1.0, 1.23, 2.0])
print('pytorch diff:', derivative_pytorch(torch.sin, x_vals))


def derivative_pytorch_autograd(x0):
    x = torch.tensor(x0, requires_grad=True)
    y = torch.sin(x)
    g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))[0]
    return g

x_vals = [0.5, 1.0, 1.23, 2.0]
print('pytorch autograd diff:', derivative_pytorch_autograd(x_vals))