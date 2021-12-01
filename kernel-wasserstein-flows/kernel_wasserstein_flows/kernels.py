import numpy as np
import torch


def gaussian_kernel(x, y, sigma):
    # XXX: explicitly ask whether we should broadcast or not?
    # axis = -1 made to be broadcastable-friendly. may not be the best solution
    if isinstance(x, torch.Tensor):
        return (-1 / sigma * (x - y) ** 2).sum(axis=-1).exp()
    else:
        return np.exp((-1 / sigma * (x - y) ** 2).sum(axis=-1))


def laplace_kernel(x, y, sigma):
    # XXX: explicitly ask whether we should broadcast or not?
    # axis = -1 made to be broadcastable-friendly. may not be the best solution
    if isinstance(x, torch.Tensor):
        return (-1 / sigma * (x - y).abs()).sum(axis=-1).exp()
    else:
        return np.exp(-1 / sigma * np.abs(x - y).sum(axis=-1))


def negative_distance_kernel(x, y, sigma):
    ret = -((x - y) ** 2).sum(axis=-1) / sigma
    return ret


def imq_kernel(x, y, sigma):
    ret = (1 ** 2 + (((x - y) ** 2) / sigma).sum(axis=-1)) ** -1
    return ret


def dot_product_squared_kernel(x, y, sigma):
    # sigma added only for compatibility - it is not used
    if isinstance(x, torch.Tensor):
        ret = torch.matmul(x, y.transpose(-1, -2))  # ** 2# + torch.dot(x, y)
        ret = ret[:, 0, :]
        return (ret ** 2) / 100
    else:
        return np.dot(x, y) ** 2 + np.dot(x, y)


def energy_kernel(x, y, sigma):
    dim = x.shape[-1]
    x0 = torch.zeros(*([1] * (len(x.shape) - 1)), dim)

    def norm_torch_sq(z):
        ret = (z ** 2).sum(axis=-1)
        return ret

    def norm_numpy_sq(z):
        return (z ** 2).sum(axis=-1)

    eps = 1e-8

    if isinstance(x, torch.Tensor):
        pxx0 = (norm_torch_sq(x - x0) + eps) ** (sigma / 2)
        pyx0 = (norm_torch_sq(y - x0) + eps) ** (sigma / 2)
        pxy = (norm_torch_sq(x - y) + eps) ** (sigma / 2)
    elif isinstance(x, np.ndarray):
        x0 = x0.detach().numpy()
        pxx0 = (norm_numpy_sq(x - x0) + eps) ** (sigma / 2)
        pyx0 = (norm_numpy_sq(y - x0) + eps) ** (sigma / 2)
        pxy = (norm_numpy_sq(x - y) + eps) ** (sigma / 2)
    else:
        raise ValueError(f"type of x ({type(x)})not understood")

    ret = 0.5 * (pxx0 + pyx0 - pxy)
    return ret


def polynomial_kernel(x, y, sigma, degree=2):
    scaling_factor = 1
    # (x.Ty + c)**d with (c=0, d=2) for now.
    # sigma is unused.
    if isinstance(x, torch.Tensor):
        # raise ValueError(x.shape, y.shape)
        ret = torch.matmul(x, y.transpose(-1, -2))  # ** 2# + torch.dot(x, y)
        ret = ret[:, 0, :] + sigma
        ret = (ret ** degree)
    else:
        assert len(y.shape) == len(x.shape) == 3
        # XXX: should use vectorized ops and not element-wise mul
        ret = ((x * y).sum(axis=-1) + sigma) ** degree
    return ret / scaling_factor


def polynomial_plus_gaussian_kernel(x, y, sigma):
    return 0.01 * polynomial_kernel(x, y, 1, degree=2) + gaussian_kernel(x, y, sigma)
