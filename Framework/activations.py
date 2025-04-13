"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""

import torch

def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + torch.exp(-z))


def tanh_activation(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return torch.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return torch.sin(z)


def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return torch.exp(-5.0 * z ** 2)


def relu_activation(z):
    return z if z > 0.0 else 0.0


def elu_activation(z):
    return z if z > 0.0 else torch.exp(z) - 1


def lelu_activation(z):
    leaky = 0.005
    return z if z > 0.0 else leaky * z


def selu_activation(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lam * z if z > 0.0 else lam * alpha * (torch.exp(z) - 1)


def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * torch.log(1 + torch.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    z = max(1e-7, z)
    return torch.log(z)


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return torch.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3