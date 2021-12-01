import types

import numpy as np
import torch

from .mmd import mmd_k

from .config import get_logger

MAX_TRAJECTORY_SNAPSHOTS = 100


def gradient_flow(
    max_iter,
    lr,
    random_seed,
    noise_level_callback,
    num_noisy_averages,
    generator,
    generator_kwargs,
    kernel,
    kernel_kwargs,
    loss,
    loss_kwargs,
    loss_first_variation,
):
    inputs = locals()

    logger = get_logger("gradient_flow")

    if isinstance(generator, (types.FunctionType, types.LambdaType)):
        X, Y = generator(**generator_kwargs)
    elif isinstance(generator, (list, tuple)):
        assert len(generator) == 2
        assert len(generator_kwargs) == 2
        generator_X, generator_Y = generator
        generator_kw_X, generator_kw_Y = generator_kwargs
        X = generator_X(**generator_kw_X)
        Y = generator_Y(**generator_kw_Y)

    logger.warning(
        f"X.requires_grad: {X.requires_grad}, "
        f"Y.requires_grad: {Y.requires_grad}"
    )

    trajectories = []
    loss_states = []

    records = []

    info = {}

    torch.manual_seed(random_seed)

    Y.requires_grad = True
    for i in range(max_iter):
        this_iter_records = {}
        noise_level = noise_level_callback(i)

        with torch.no_grad():
            loss_val, info, loss_records = loss(
                X, Y, kernel, kernel_kwargs, last_iter_info=info, **loss_kwargs
            )

            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()

        if num_noisy_averages > 1:
            U = noise_level * torch.randn(
                num_noisy_averages, *Y.shape, requires_grad=False
            )
            # broadcasted operation
            _grad_eval_points = Y + U
        elif num_noisy_averages == 1:
            U = noise_level * torch.randn(*Y.shape, requires_grad=False)
            _grad_eval_points = Y + U
        elif num_noisy_averages == 0:
            _grad_eval_points = Y
        else:
            raise ValueError

        first_variation_val = loss_first_variation(
            X.detach(), Y.detach(), _grad_eval_points, kernel, kernel_kwargs, info
        )
        first_variation_val.backward()

        # noise injection: evaluate gradient of first variation at
        # x + u: grad_x(f o (x -> x + u))(x) =  grad_x(f)(x + u)
        with torch.no_grad():
            Y -= lr * Y.grad

        # loss_vals.append(loss.detach().numpy().copy())
        this_iter_records["loss"] = loss_val
        this_iter_records["noise_level"] = noise_level
        this_iter_records["learning_rate"] = lr

        with torch.no_grad():
            this_iter_records["grad_norm"] = (Y.grad ** 2).sum().sqrt().item()
        Y.grad.zero_()

        assert all(r not in this_iter_records for r in loss_records)
        this_iter_records.update(loss_records)

        records.append(this_iter_records)

        if i % max(max_iter // 10, 1) == 0:
            msg = " ".join(
                [f"{k}: {v:.2e}" for k, v in this_iter_records.items()
                 if k not in ('noise_level', 'learning_rate')]
            )
            logger.info(msg)

        if i % max(max_iter // MAX_TRAJECTORY_SNAPSHOTS, 1) == 0:
            # Trajectories take a lot of memory, dont record them at every
            # steps
            trajectories.append(Y.detach().clone())
            loss_states.append(
                {**info,
                 # 'grad': noisy_Y.grad.detach().clone().numpy()
                }
            )

    trajectories = torch.stack(trajectories, axis=0).detach().clone().numpy()

    return (
        inputs,
        (X.detach().numpy(), Y.detach().numpy()),
        (trajectories, records, loss_states),
    )
