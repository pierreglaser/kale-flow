import numpy as np
import types
import torch


def unadjusted_langevin_algorithm(
    max_iter, lr, random_seed, generator, generator_kwargs
):
    inputs = locals()
    if isinstance(generator, types.FunctionType):
        X, Y, potential = generator(**generator_kwargs)
    elif isinstance(generator, (list, tuple)):
        assert len(generator) == 2
        assert len(generator_kwargs) == 2
        generator_X, generator_Y = generator
        generator_kw_X, generator_kw_Y = generator_kwargs
        X, potential = generator_X(**generator_kw_X)
        Y = generator_Y(**generator_kw_Y)

    step_size = lr
    torch.manual_seed(random_seed)
    # trajectory = [Y.detach().numpy().copy()]
    trajectory = []

    for epoch in range(max_iter):
        _loss = potential(Y)
        _loss.backward()

        with torch.no_grad():
            Y += -step_size * Y.grad + np.sqrt(2 * step_size) * torch.randn(
                Y.shape
            )
        Y.grad.zero_()
        trajectory.append(Y.detach().numpy().copy())
    trajectory = np.array(trajectory)
    return (
        inputs,
        (X.detach().numpy(), Y.detach().numpy()),
        (trajectory, {}, {}),
    )
