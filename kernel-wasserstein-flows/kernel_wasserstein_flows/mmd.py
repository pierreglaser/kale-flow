import torch

import numpy as np

from .config import get_logger


SIGMAS = list(map(lambda x: 1.5 ** x, list(range(-20, 20))))


def mmd(x, y, kernel, kernel_kwargs, last_iter_info):
    # it's 1/2 * MMD ** 2 really...
    assert len(x.shape) == len(y.shape) == 2
    assert x.shape[1] == y.shape[1]
    N, M = x.shape[0], y.shape[0]

    kxy = kernel(x[:, None, :], y[None, :, :], **kernel_kwargs)
    kxx = kernel(x[:, None, :], x[None, :, :], **kernel_kwargs)
    kyy = kernel(y[:, None, :], y[None, :, :], **kernel_kwargs)

    return (
        0.5 * (kxx.sum() / N ** 2 + kyy.sum() / M ** 2 - 2 * kxy.sum() / (N * M)),  # noqa
        {},
        {},
    )


def mmd_k(x, y, U, kernel, kernel_kwargs, info):
    assert not x.requires_grad
    assert not y.requires_grad
    assert not U.requires_grad

    assert len(x.shape) == len(y.shape) == 2

    if len(U.shape) == 3:
        kxex = kernel(
            x[:, None, None, :], x + U[None, :, :], **kernel_kwargs
        ).mean()
        kyey = kernel(
            y[:, None, None, :], y + U[None, :, :], **kernel_kwargs
        ).mean()
        kyex = kernel(
            y[:, None, None, :], x + U[None, :, :], **kernel_kwargs
        ).mean()
        kxey = kernel(
            x[:, None, None, :], y + U[None, :, :], **kernel_kwargs
        ).mean()
        return 1 / 2 * (kxex + kyey - kyex - kxey)
    elif len(U.shape) == 2:
        ret, _, _ = mmd(x, y, kernel, kernel_kwargs, info)
        return ret
    else:
        raise ValueError


def mmd_first_variation(
    x, y, eval_pts, kernel, kernel_kwargs, info, strict=True
):

    if strict:
        assert not x.requires_grad
        assert not y.requires_grad
        assert eval_pts.requires_grad

        assert len(x.shape) == len(y.shape) == 2

    if len(eval_pts.shape) == 3:
        kxz = kernel(
            x[:, None, None, :], eval_pts[None, :, :], **kernel_kwargs
        )
        kyz = kernel(
            y[:, None, None, :], eval_pts[None, :, :], **kernel_kwargs
        )
    elif len(eval_pts.shape) == 2:
        kxz = kernel(x[:, None, :], eval_pts[None, :, :], **kernel_kwargs)

        kyz = kernel(y[:, None, :], eval_pts[None, :, :], **kernel_kwargs)
    else:
        raise ValueError

    return kyz.sum() / y.shape[0] - kxz.sum() / x.shape[0]


def find_optimal_kernel_width_grid_search(x, y, kernel, width_grid):
    logger = get_logger("mmd.kernel_optimization")
    assert x.requires_grad
    assert not y.requires_grad
    mmd_vals = []
    grads = []
    grads_of_grads_norms = []

    if x.grad is not None:
        x.grad.zero_()
    for width in width_grid:
        # XXX: only works with sigma-based kernels for now.
        kernel_kwargs = {"sigma": width}

        val, _, _ = mmd(
            x, y, kernel=kernel, kernel_kwargs=kernel_kwargs, last_iter_info={}
        )
        mmd_vals.append(val)

        eval_pts = torch.from_numpy(y.detach().numpy())
        eval_pts.requires_grad = True

        first_var = mmd_first_variation(
            x, y, eval_pts, kernel, kernel_kwargs, {}, strict=False
        )

        (grad_,) = torch.autograd.grad(first_var, eval_pts, create_graph=True)

        # use froebenius norm, probably should use some operator norm
        grad_norm = (grad_ ** 2).sum()
        grad_norm.backward()

        grad_of_grad = x.grad.detach().numpy()
        grad_of_grad_norm = (grad_of_grad ** 2).sum()

        grads_of_grads_norms.append(grad_of_grad_norm)
        grads.append(grad_)
        logger.debug(f"sigma: {width} grad of grad norm: {grad_of_grad_norm}")
        x.grad.zero_()
        eval_pts.grad.zero_()

    grads_of_grads_norms = np.array(grads_of_grads_norms)
    max_idx = np.argmax(grads_of_grads_norms)
    return (max_idx, (mmd_vals, width_grid, grads, grads_of_grads_norms))


def find_optimal_kernel_width_gradient_descent(
    x, y, kernel, sigma_0, lr_0, max_iter, clip_value
):
    logger = get_logger("mmd.kernel_optimization")
    assert x.requires_grad
    assert not y.requires_grad
    mmd_vals = []

    if x.grad is not None:
        x.grad.zero_()

    sigma_0 = float(sigma_0)
    lr = float(lr_0)

    sigma = torch.tensor([sigma_0])[0]
    sigma.requires_grad = True
    logger.debug(f"INIT: sigma_0: {sigma_0:.2E}, lr: {lr_0:.2E}")

    gain = 0

    for i in range(max_iter):
        # XXX: only works with sigma-based kernels for now.
        kernel_kwargs = {"sigma": sigma}

        val, _, _ = mmd(
            x, y, kernel=kernel, kernel_kwargs=kernel_kwargs, last_iter_info={}
        )
        mmd_vals.append(val)

        eval_pts = torch.from_numpy(y.detach().numpy())
        eval_pts.requires_grad = True

        eval_pts_2 = torch.from_numpy(x.detach().numpy())
        eval_pts_2.requires_grad = True

        first_var = mmd_first_variation(
            x, y, eval_pts, kernel, kernel_kwargs, {}, strict=False
        )

        (grad_,) = torch.autograd.grad(first_var, eval_pts, create_graph=True)

        # use froebenius norm, probably should use some operator norm
        grad_norm = (grad_ ** 2).sum()

        (grad_wrt_target,) = torch.autograd.grad(
            grad_norm, x, create_graph=True
        )
        prev_gain = gain

        if clip_value >= 0:
            grad_wrt_target = grad_wrt_target.clamp(-clip_value, clip_value)

        grad_wrt_target_norm = (grad_wrt_target ** 2).sum()

        gain = grad_wrt_target_norm

        penalize_close_to_target = False

        if penalize_close_to_target:
            first_var_2 = mmd_first_variation(
                x, y, eval_pts_2, kernel, kernel_kwargs, {}, strict=False
            )

            (grad_2,) = torch.autograd.grad(
                first_var_2, eval_pts_2, create_graph=True
            )
            grad_norm_2 = (grad_2 ** 2).sum() / 1e4
            gain += grad_norm_2

        (gain).backward()

        delta_sigma = lr * sigma.grad

        next_sigma_negative = (sigma + delta_sigma) < 0
        objective_not_increasing = prev_gain > gain

        if next_sigma_negative or (objective_not_increasing):
            logger.info(
                f"iter : {i}: decreasing LR: {lr} (reason: sigma < 0 or "
                f"objective decreasing)"
            )
            lr /= 2

            x.grad.zero_()
            sigma.grad.zero_()
            eval_pts.grad.zero_()

            continue

        if sigma > 1000:
            logger.warning(
                f"sigma is blowing up: sigma: {sigma}, delta_sigma: "
                f"{delta_sigma}, iter: {i}"
            )

        if i % max(max_iter // 10, 1) == 0:
            if penalize_close_to_target:
                logger.debug(
                    f"iter_no: {i}, sigma: {sigma:.2E}, "
                    f"grad_wrt_target: {grad_wrt_target_norm:.2E}, "
                    f"grad_close_to_target: {grad_norm_2:.2E} "
                )
            else:
                logger.debug(
                    f"iter_no: {i}, sigma: {sigma:.2E}, "
                    f"grad_wrt_target: {grad_wrt_target_norm:.2E}"
                )
        with torch.no_grad():
            sigma += delta_sigma

        x.grad.zero_()
        eval_pts.grad.zero_()
        sigma.grad.zero_()

        if delta_sigma.abs() < 1e-5:
            break

        if i == (max_iter - 1):
            pass
            # logger.warning("did not converge")

    logger.debug(
        f"END: iter_no: {i}, lr: {lr:.2E}, sigma: {sigma:.2E}, "
        f"delta_sigma: {delta_sigma:.2E}"
    )
    return (sigma.detach().numpy(), val, lr, i)


def mmd_optimal_width_grid_search(
    x, y, kernel, kernel_kwargs, last_iter_info, grid
):
    assert len(kernel_kwargs) == 0
    (
        max_idx,
        (mmd_vals, width_grid, grads, grads_of_grads_norm),
    ) = find_optimal_kernel_width_grid_search(x, y, kernel, grid)

    return (
        mmd_vals[max_idx],
        {"grad": grads[max_idx], "sigma": width_grid[max_idx]},
        {"sigma": width_grid[max_idx], "DvDx": grads_of_grads_norm[max_idx]},
    )


def mmd_optimal_width_gradient_descent(
    x,
    y,
    kernel,
    kernel_kwargs,
    last_iter_info,
    sigma_0,
    lr_0,
    max_iter,
    clip_value,
):
    sigma_0 = last_iter_info.get("sigma", sigma_0)
    # lr_0 = last_iter_info.get("lr", lr_0)

    sigma, mmd_val, lr, num_iter = find_optimal_kernel_width_gradient_descent(
        x,
        y,
        kernel,
        sigma_0=sigma_0,
        lr_0=lr_0,
        max_iter=max_iter,
        clip_value=clip_value,
    )
    sigma = sigma[()]
    return (
        mmd_val,
        {"sigma": sigma, "lr": lr},
        {"sigma": sigma, "lr": lr, "num_iter": num_iter},
    )


def mmd_optimal_width_first_variation(
    x, y, eval_pts, kernel, kernel_kwargs, info, strict=True
):
    sigma = info["sigma"]
    return mmd_first_variation(
        x, y, eval_pts, kernel, {"sigma": sigma}, {}, strict=False
    )


def profile_mmd_across_trajectory(X, trajectories, kernel, width_grid):
    all_mmd_vals, all_grads_of_grads_norms = [], []
    for i in range(len(trajectories)):
        (max_idx, (mmd_vals, width_grid, grads, grads_of_grads_norms)) = find_optimal_kernel_width_grid_search(
            X,
            torch.from_numpy(trajectories[i]),
            kernel,
            width_grid=width_grid
        )

        mmd_vals = np.array([m.detach().numpy() for m in mmd_vals])
        all_mmd_vals.append(mmd_vals)
        all_grads_of_grads_norms.append(grads_of_grads_norms)
    return all_mmd_vals, all_grads_of_grads_norms
