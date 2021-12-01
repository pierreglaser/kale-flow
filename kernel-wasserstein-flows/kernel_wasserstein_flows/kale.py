import time
import functools

import torch
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.special import lambertw

from .config import get_logger


def _unpenalized_kale_dual(alpha, N):
    return 1 + np.sum(alpha * np.log(alpha)) + np.sum(alpha) * np.log(N / np.e)


def _kale_dual(alpha, K_xx, K_xy, K_yy, lambda_, kernel_kwargs, penalized):
    Nx = K_xx.shape[0]
    Ny = K_yy.shape[0]

    N = len(alpha)  # should be == Nx
    assert N == Nx

    tol = 1e-50
    ret = np.sum(alpha * np.log(tol + Nx * alpha)) - np.sum(alpha)

    xx_part = (np.outer(alpha, alpha) * K_xx).sum()
    xy_part = (K_xy.T @ alpha).sum() / Ny
    yy_part = K_yy.sum() / (Ny ** 2)

    if penalized:
        norm_squared = 1 / (2 * lambda_) * (xx_part - 2 * xy_part + yy_part)
        ret += norm_squared
        return -ret
    else:
        return -ret


def _kale_primal(
    alpha,
    K_xx,
    K_xy,
    K_yy,
    lambda_,
    kernel_kwargs,
    penalized,
    also_return_norm_term,
):
    K_yx = K_xy.T
    Ny = K_yy.shape[0]
    Nx = K_xx.shape[0]

    primal = (
        np.exp(
            1 / lambda_ * (K_xy @ (1 / Ny * np.ones(Ny)) - K_xx @ alpha)
        ).sum()
        / Nx
        - (
            +1 / lambda_ * (K_yy @ (1 / Ny * (np.ones(Ny)))).sum()
            - 1 / lambda_ * (K_yx @ alpha).sum()
        )
        / Ny
    )

    if also_return_norm_term or penalized:
        norm_squared = (
            1
            / (2 * lambda_)
            * (
                (np.outer(alpha, alpha) * K_xx).sum()
                - 2 / Ny * (K_yx @ alpha).sum()
                + 1 / (Ny ** 2) * K_yy.sum()
            )
        )
        if penalized:
            primal += norm_squared

    if also_return_norm_term:
        return primal, norm_squared
    else:
        return primal


def dual_kale_objective(
    alpha: np.ndarray,
    K_xy: np.ndarray,
    K_xx: np.ndarray,
    lambda_: float,
    input_check: bool,
):
    """
    Objective function whose maximizer yields the alpha used to compute KALE.
    Strictly equal to _penalized_kale_dual modulo signs and constants.

    sum(alpha_i * log(N * alpha_i) - alpha-i) +
        1/(2 * lambda_) * norm_rkhs(sum(alpha_iK_xi - 1/n K_yi))
    """
    if input_check:
        assert K_xx.shape[0] == K_xx.shape[1]
        assert np.allclose(
            K_xx, K_xx.T, rtol=1e-5, atol=1e-8
        ), "K_xx must be symmetric"

    Ny = K_xy.shape[1]
    Nx = K_xy.shape[0]

    # expanded rkhs norm
    neg_kl = alpha.T @ K_xx @ alpha / (2 * lambda_)
    neg_kl -= np.sum(alpha.T @ K_xy) / (lambda_ * Ny)

    # sum(x log x)
    neg_kl += np.sum(np.log(Nx * alpha) * alpha)

    # sum(x) log n - sum(x)
    neg_kl -= np.sum(alpha)
    return neg_kl


def grad_dual_kale_obj(alpha, K_xx, K_xy: np.ndarray, lambda_, rescale=False):
    Nx, Ny = K_xy.shape

    _grad_norm = K_xx @ alpha - 1 / Ny * K_xy @ np.ones(Ny)
    ret = np.log(Nx * alpha) + 1 / lambda_ * _grad_norm
    if rescale:
        return lambda_ * ret
    else:
        return ret


def hess_dual_kale_obj(alpha, K_xx, lambda_, rescale=False):
    _diag = 1 / (alpha)
    if np.any(_diag == np.inf):
        raise ValueError
    if rescale:
        r1 = lambda_ * np.diag(_diag)
        r2 = K_xx
        return r1 + r2
    else:
        r1 = np.diag(_diag)
        r2 = 1 / lambda_ * K_xx
        return r1 + r2


def newton_method(
    alpha_0: np.ndarray,
    K_xx: np.ndarray,
    K_xy: np.ndarray,
    lambda_: float,
    max_iter: int,
    a: float,
    b: float,
    inplace: bool,
    tol: float,
    input_check: bool,
):
    """
    Gradient descent algorithm with line search and positivity constraints

    This algorithm is specific to the dual kale objective function.
    """
    t0 = time.time()
    logger = get_logger("kale.optim.newton")

    if not inplace:
        alpha_0 = alpha_0.copy()

    alpha = alpha_0

    max_line_search_iter = 0

    for i in range(max_iter):
        j_val = dual_kale_objective(
            alpha=alpha,
            K_xy=K_xy,
            K_xx=K_xx,
            lambda_=lambda_,
            input_check=input_check,
        )
        grad_j_val = grad_dual_kale_obj(
            alpha=alpha, K_xx=K_xx, K_xy=K_xy, lambda_=lambda_
        )

        try:
            hess_j_val = hess_dual_kale_obj(alpha, K_xx, lambda_)
        except ValueError:
            msg = (
                f"overflow at iteration {i} while computing kale "
                f" using newton's method. This is likely due to "
                f"lambda being too low"
            )
            logger.critical(msg)
            raise ValueError(msg)

        inv_hess = np.linalg.inv(hess_j_val)

        delta = inv_hess @ grad_j_val

        newton_decrement = grad_j_val @ inv_hess @ grad_j_val
        if newton_decrement / 2 < tol:
            break

        t, num_iter_line_search = line_search(
            alpha=alpha,
            a=a,
            b=b,
            delta=-delta,
            lambda_=lambda_,
            grad_f=grad_j_val,
            K_xy=K_xy,
            K_xx=K_xx,
            J_init=j_val,
            input_check=input_check,
        )

        max_line_search_iter = max(max_line_search_iter, num_iter_line_search)

        if i % (max_iter // 10) == 0:
            logger.debug(
                f"iter_no, {i}, f(alpha), {j_val:.4f} "
                f"grad norm, {np.linalg.norm(grad_j_val):e}, "
                f"n_iter_line_search: {num_iter_line_search}, "
                # f"t_line_search: {t:.5f}"
            )

        alpha += t * (-delta)
        # alpha -= t * grad_j_val
    else:
        logger.warning(f"Newton method did not converge after {i} iterations")

    logger.info(
        f"{i} iterations, "
        f"total time: {time.time() - t0:.2f}s, "
        f"max line search steps: {max_line_search_iter}, "
        f"stopping criterion  {np.linalg.norm(newton_decrement / 2):.2e}"
    )
    return (alpha, {})


def line_search(
    alpha: np.array,
    a: float,
    b: float,
    delta: float,
    lambda_: float,
    grad_f: np.ndarray,
    K_xy: np.ndarray,
    K_xx: np.ndarray,
    J_init: float,
    input_check: bool,
):
    logger = get_logger("kale.line_search")
    t = 1.0

    if np.all(alpha + t * delta >= 0):
        f_next = dual_kale_objective(
            alpha=alpha + t * delta,
            K_xy=K_xy,
            K_xx=K_xx,
            lambda_=lambda_,
            input_check=input_check,
        )
        f_th = J_init + a * t * grad_f @ delta

    i = 0
    found_feasible = False
    while not found_feasible or (i < 10 and np.isnan(f_next) or f_next > f_th):
        t *= b
        if np.all(alpha + t * delta >= 0):
            if not found_feasible:
                found_feasible = True
                logger.debug(f"found feasible point after {i} iter")
            f_next = dual_kale_objective(
                alpha=alpha + t * delta,
                K_xy=K_xy,
                K_xx=K_xx,
                lambda_=lambda_,
                input_check=input_check,
            )
            f_th = J_init + a * t * grad_f @ delta
        else:
            found_feasible = False
        i += 1

    assert found_feasible

    logger.debug(
        f"final f {f_next:.3e}, minimal descent {f_th:.3e} final t {t}, "
        f"iter {i}"
    )
    return t, i


def kale(
    X,
    Y,
    kernel,
    kernel_kwargs,
    lambda_,
    inner_max_iter,
    inner_a,
    inner_b,
    inner_tol,
    inplace,
    input_check,
    last_iter_info,
    penalized,
    allow_primal_overflow=False,
    optimization_method="newton",
    optimizer_kwargs=None,
    online=False,
    dual_gap_tol=1e-4,
):
    assert optimization_method in ("newton", "l-bfgs", "cd")
    if optimization_method in ("l-bfgs", "newton"):
        errmsg = "l-bfgs and newton do not support online method for now"
        assert not online, errmsg

    # allow_primal_overflow: the primal formulation of KALE involves an exp
    # term that is prone to numerical unstability. Setting
    # allow_primal_overflow to True will skip the dual gap check, which in
    # turns implies that the primal formulation of KALE is never computed.
    # TODO: implement it

    # inner_a, inner_b, inner_tol, inner_max_iter are now deprecated in favour
    # of optimizer_kwargs

    # move to numpy mode
    X = X.detach().numpy()
    Y = Y.detach().numpy()

    if not online:
        K_xx = kernel(X[:, None, :], X[None, :, :], **kernel_kwargs)
        K_xy = kernel(X[:, None, :], Y[None, :, :], **kernel_kwargs)
        K_yy = kernel(Y[:, None, :], Y[None, :, :], **kernel_kwargs)

    if "alpha" in last_iter_info:
        # Warm start using solution of last iteration
        alpha_0 = last_iter_info["alpha"].copy()
    else:
        alpha_0 = 0.001 * np.ones(X.shape[0])

    primal = dual = norm_term = None
    if optimization_method == "newton":
        opt_kwargs = optimizer_kwargs
        if opt_kwargs is None:
            # backward compat for old, argument-per-argument way of passing opt
            # kwargs
            opt_kwargs = dict(
                max_iter=inner_max_iter, a=inner_a, b=inner_b, tol=inner_tol,
            )

        alpha, kale_estimation_info = newton_method(
            alpha_0=alpha_0,
            K_xx=K_xx,
            K_xy=K_xy,
            lambda_=lambda_,
            inplace=inplace,
            input_check=input_check,
            **opt_kwargs,
        )

    elif optimization_method == "l-bfgs":
        _func = functools.partial(
            dual_kale_objective,
            K_xy=K_xy,
            K_xx=K_xx,
            lambda_=lambda_,
            input_check=input_check,
        )
        _grad = functools.partial(
            grad_dual_kale_obj, K_xx=K_xx, K_xy=K_xy, lambda_=lambda_
        )
        opt_kwargs = dict(
            m=100,
            # factr=1,
            factr=100,
            # pgtol=3e-2,
            pgtol=1e-7,
            iprint=0,
            maxfun=15000,
            maxiter=50,
            disp=0,
            callback=None,
            maxls=20,
        )
        if optimizer_kwargs is not None:
            for k, v in optimizer_kwargs.items():
                opt_kwargs[k] = v

        alpha, _, _ = fmin_l_bfgs_b(
            _func,
            alpha_0,
            fprime=_grad,
            args=(),
            bounds=[(1e-8, None) for _ in range(len(alpha_0))],
            **opt_kwargs,
        )
        # TODO (pierreglaser): log some l-bfgs metrics in kale_estimation_info?
        kale_estimation_info = {}

    elif optimization_method == "cd":
        opt_kwargs = dict(max_iter=50, dual_gap_tol=1e-8)
        if optimizer_kwargs is not None:
            for k, v in optimizer_kwargs.items():
                opt_kwargs[k] = v

        if optimizer_kwargs is not None:
            for k, v in optimizer_kwargs.items():
                opt_kwargs[k] = v
        if online:
            # hardcoded buffer size of 1000 x 1000 has good performance
            if "buffer_size" not in opt_kwargs:
                opt_kwargs["buffer_size"] = 1000
            ret = online_kale_coordinate_descent(
                alpha_0=alpha_0,
                lambda_=lambda_,
                X=X,
                Y=Y,
                kernel=kernel,
                kernel_kwargs=kernel_kwargs,
                **opt_kwargs,
            )
            alpha, dual, primal, norm_term, kale_estimation_info = ret
        else:
            alpha, kale_estimation_info = kale_coordinate_descent(
                alpha_0=alpha_0,
                lambda_=lambda_,
                K_xx=K_xx,
                K_xy=K_xy,
                K_yy=K_yy,
                **opt_kwargs,
            )

    if primal is None or dual is None or norm_term is None:
        assert not online
        assert primal is dual is norm_term is None
        primal, norm_term = _kale_primal(
            alpha,
            K_xx,
            K_xy,
            K_yy,
            lambda_,
            kernel_kwargs,
            penalized=True,
            also_return_norm_term=True,
        )
        dual = _kale_dual(
            alpha, K_xx, K_xy, K_yy, lambda_, kernel_kwargs, penalized=True
        )

    # make sure the dual gap is reasonably small
    absolute_gap = np.abs(primal - dual)
    relative_gap = np.abs(primal - dual) / min(np.abs(primal), np.abs(dual))
    info = {"primal": primal, "dual": dual, "norm_term": norm_term}
    if absolute_gap > dual_gap_tol and relative_gap > dual_gap_tol:
        msg = (
            f"dual gap too high after kale optimization: "
            f"absolute dual gap: {absolute_gap}, "
            f"relative dual gap: {relative_gap}, {info}"
        )
        raise ValueError(msg)

    extra_callbacks = {
        **kale_estimation_info,
    }

    if not penalized:
        primal -= norm_term

    # alpha /= (1 * np.sum(alpha))
    # print(np.sum(alpha))
    return (
        (1 + lambda_) * (1 - primal),
        {"alpha": alpha, "lambda_": lambda_},
        extra_callbacks,
    )


def kale_penalized(
    X,
    Y,
    kernel,
    kernel_kwargs,
    lambda_,
    inner_max_iter,
    inner_a,
    inner_b,
    inner_tol,
    inplace,
    input_check,
    last_iter_info,
    allow_primal_overflow=False,
    optimization_method="newton",
    optimizer_kwargs=None,
    online=False,
    dual_gap_tol=1e-4,
):
    return kale(
        X,
        Y,
        kernel,
        kernel_kwargs,
        lambda_,
        inner_max_iter,
        inner_a,
        inner_b,
        inner_tol,
        inplace,
        input_check,
        last_iter_info,
        penalized=True,
        allow_primal_overflow=allow_primal_overflow,
        optimization_method=optimization_method,
        optimizer_kwargs=optimizer_kwargs,
        online=online,
        dual_gap_tol=dual_gap_tol,
    )


def kale_unpenalized(
    X,
    Y,
    kernel,
    kernel_kwargs,
    lambda_,
    inner_max_iter,
    inner_a,
    inner_b,
    inner_tol,
    inplace,
    input_check,
    last_iter_info,
    allow_primal_overflow=False,
    optimization_method="newton",
    optimizer_kwargs=None,
    online=False,
    dual_gap_tol=1e-4,
):
    return kale(
        X,
        Y,
        kernel,
        kernel_kwargs,
        lambda_,
        inner_max_iter,
        inner_a,
        inner_b,
        inner_tol,
        inplace,
        input_check,
        last_iter_info,
        penalized=False,
        allow_primal_overflow=allow_primal_overflow,
        optimization_method=optimization_method,
        optimizer_kwargs=optimizer_kwargs,
        online=online,
        dual_gap_tol=dual_gap_tol,
    )


def kale_penalized_first_variation(
    x: torch.Tensor,
    y: torch.Tensor,
    eval_pts: torch.Tensor,
    kernel,
    kernel_kwargs,
    info,
):
    alpha = info["alpha"]
    lambda_ = info["lambda_"]

    # KALE(P || Q) depends on P like w dP + ...
    # -> grad is w = 1/n K_Y @ 1 - K_X @ alpha
    # assert not x.requires_grad
    # assert not y.requires_grad

    Nx = x.shape[0]  # noqa
    Ny = y.shape[0]
    # assert eval_pts.requires_grad

    # In the standard KALE, the ys (resp. xs) are assumed to be sampled from P
    # (resp. Q)

    assert len(x.shape) == len(y.shape)
    if len(eval_pts.shape) == 2:
        kzx = kernel(eval_pts[:, None, :], x[None, :, :], **kernel_kwargs)
        kzy = kernel(eval_pts[:, None, :], y[None, :, :], **kernel_kwargs)

        ret1 = kzy.sum() / Ny
        ret2 = (kzx @ torch.from_numpy(alpha).float()).sum()

        w = ret1 - ret2
        return (1 + lambda_) * w / lambda_

    elif len(eval_pts.shape) == 3:
        # X dim: (n, d)
        # eval_pts dim: (k, n, d)
        # output of the kernel: (k, n, n)
        # the k dimension should be placed first to maintain a format
        # compatible with the matmul call afterwards (in matmul, the reduced
        # dimensions should be placed last)
        kzy = kernel(
            eval_pts[:, :, None, :], y[None, None, :, :], **kernel_kwargs
        )
        kzx = kernel(
            eval_pts[:, :, None, :], x[None, None, :, :], **kernel_kwargs
        )
        ret1 = kzy.sum() / Ny
        ret2 = torch.matmul(kzx, torch.from_numpy(alpha).float()).sum()

        # XXX: why dividing by eval_points?? should not be divided
        w = (ret1 - ret2) / eval_pts.shape[0]
        return (1 + lambda_) * w / lambda_
    else:
        raise ValueError("eval_pts should have 2 or 3 dimensions")


def kale_unpenalized_first_variation(
    x: torch.Tensor,
    y: torch.Tensor,
    eval_pts: torch.Tensor,
    kernel,
    kernel_kwargs,
    info,
):
    assert not x.requires_grad
    assert not y.requires_grad
    # assert eval_pts.requires_grad
    assert len(x.shape) == len(y.shape)

    alpha = info["alpha"]
    lambda_ = info["lambda_"]
    n = len(x)
    m = len(y)

    K_xy_xy = kernel(
        torch.cat((x, y), axis=0)[:, None, :],
        torch.cat((x, y), axis=0)[None, :, :],
        **kernel_kwargs,
    )
    K_xx = K_xy_xy[:n, :n]
    K_xy = K_xy_xy[:n, n:]

    hx = (
        1 / m * K_xy.sum(axis=1) - K_xx @ torch.from_numpy(alpha).float()
    ) / lambda_

    D = 1 / n * torch.diag(torch.cat((torch.exp(hx), torch.zeros(m))))

    v = torch.cat((1 / n * torch.exp(hx), -1 / m * torch.ones(m)))

    coefs = torch.inverse(D @ K_xy_xy + lambda_ * torch.eye(n + m)) @ v

    if len(eval_pts.shape) == 2:
        k_z_xy = kernel(
            eval_pts[:, None, :],
            torch.cat((x, y), axis=0)[None, :, :],
            **kernel_kwargs,
        )
        ret = (k_z_xy @ coefs).sum()
    elif len(eval_pts.shape) == 3:
        k_z_xy = kernel(
            eval_pts[:, :, None, :],
            torch.cat((x, y), axis=0)[None, None, :, :],
            **kernel_kwargs,
        )
        ret = (k_z_xy @ coefs).sum() / eval_pts.shape[0]
    else:
        raise ValueError("eval_pts should have 2 or 3 dimensions")

    penalized_grad = kale_penalized_first_variation(
        x, y, eval_pts, kernel, kernel_kwargs, info
    )
    return penalized_grad - (1 + lambda_) * ret


def reverse_kale_penalized(
    X,
    Y,
    kernel,
    kernel_kwargs,
    lambda_,
    inner_max_iter,
    inner_a,
    inner_b,
    inner_tol,
    inplace,
    input_check,
    last_iter_info,
    allow_primal_overflow=False,
    optimization_method="newton",
    optimizer_kwargs=None,
    online=False,
    dual_gap_tol=1e-4,
):
    # We want to compute KALE(P || Q), but here, Q is moving (in the standard
    # KALE, P is moving). The usual convention is to have y ~ P, but right now,
    # since the moving point cloud is always y, y ~ Q.  To restore the correct
    # convention, we thus must swap the x and y.
    return kale_penalized(
        Y,
        X,
        kernel,
        kernel_kwargs,
        lambda_,
        inner_max_iter,
        inner_a,
        inner_b,
        inner_tol,
        inplace,
        input_check,
        last_iter_info,
        allow_primal_overflow=allow_primal_overflow,
        optimization_method=optimization_method,
        optimizer_kwargs=optimizer_kwargs,
        online=online,
        dual_gap_tol=dual_gap_tol,
    )


def reverse_kale_unpenalized(
    X,
    Y,
    kernel,
    kernel_kwargs,
    lambda_,
    inner_max_iter,
    inner_a,
    inner_b,
    inner_tol,
    inplace,
    input_check,
    last_iter_info,
    allow_primal_overflow=False,
    optimization_method="newton",
    optimizer_kwargs=None,
    online=False,
    dual_gap_tol=1e-4,
):
    # We want to compute KALE(P || Q), but here, Q is moving (in the standard
    # KALE, P is moving). The usual convention is to have y ~ P, but right now,
    # since the moving point cloud is always y, y ~ Q.  To restore the correct
    # convention, we thus must swap the x and y.
    return kale_unpenalized(
        Y,
        X,
        kernel,
        kernel_kwargs,
        lambda_,
        inner_max_iter,
        inner_a,
        inner_b,
        inner_tol,
        inplace,
        input_check,
        last_iter_info,
        allow_primal_overflow=allow_primal_overflow,
        optimization_method=optimization_method,
        optimizer_kwargs=optimizer_kwargs,
        online=online,
        dual_gap_tol=dual_gap_tol,
    )


def reverse_kale_penalized_first_variation(
    x: torch.Tensor,
    y: torch.Tensor,
    eval_pts: torch.Tensor,
    kernel,
    kernel_kwargs,
    info,
):
    alpha = info["alpha"]
    lambda_ = info["lambda_"]

    # KALE(P || Q) depends on Q like - int (exp(h) dQ)
    # w = 1/n K_Y @ 1 - K_X @ alpha
    assert not x.requires_grad
    assert not y.requires_grad
    # assert eval_pts.requires_grad

    # We want to compute KALE(P || Q), but here, Q is moving (in the standard
    # KALE, P is moving). The usual convention is to have y ~ P, but right now,
    # since the moving point cloud is always y, y ~ Q.  To restore the correct
    # convention, we thus must swap the x and y.
    _int = x
    x = y
    y = _int

    assert len(x.shape) == len(y.shape)
    if len(eval_pts.shape) == 2:
        kzx = kernel(eval_pts[:, None, :], x[None, :, :], **kernel_kwargs)
        kzy = kernel(eval_pts[:, None, :], y[None, :, :], **kernel_kwargs)

        ret1 = 1 / y.shape[0] * kzy.sum(axis=-1)
        ret2 = kzx @ torch.from_numpy(alpha).float()

        ret = -((ret1 - ret2) / lambda_).exp().sum()
        return (1 + lambda_ ) * ret

    elif len(eval_pts.shape) == 3:
        # X dim: (n, d)
        # eval_pts dim: (k, n, d)
        # output of the kernel: (k, n, n)
        # the k dimension should be placed first to maintain a format
        # compatible with the matmul call afterwards (in matmul, the reduced
        # dimensions should be placed last)
        kzy = kernel(
            eval_pts[:, :, None, :], y[None, None, :, :], **kernel_kwargs
        )
        kzx = kernel(
            eval_pts[:, :, None, :], x[None, None, :, :], **kernel_kwargs
        )

        ret1 = 1 / y.shape[0] * kzy.sum(axis=-1)
        ret2 = torch.matmul(kzx, torch.from_numpy(alpha).float())

        ret = -((ret1 - ret2) / lambda_).exp().sum()

        return (1 + lambda_) * ret / eval_pts.shape[0]
    else:
        raise ValueError("eval_pts should have 2 or 3 dimensions")


def reverse_kale_unpenalized_first_variation(
    x: torch.Tensor,
    y: torch.Tensor,
    eval_pts: torch.Tensor,
    kernel,
    kernel_kwargs,
    info,
):
    assert not x.requires_grad
    assert not y.requires_grad
    # assert eval_pts.requires_grad
    assert len(x.shape) == len(y.shape)

    alpha = info["alpha"]
    lambda_ = info["lambda_"]

    penalized_grad = reverse_kale_penalized_first_variation(
        x, y, eval_pts, kernel, kernel_kwargs, info
    )

    # We want to compute KALE(P || Q), but here, Q is moving (in the standard
    # KALE, P is moving). The usual convention is to have y ~ P, but right now,
    # since the moving point cloud is always y, y ~ Q.  To restore the correct
    # convention, we thus must swap the x and y.
    _int = x
    x = y
    y = _int

    n = len(x)
    m = len(y)

    K_xy_xy = kernel(
        torch.cat((x, y), axis=0)[:, None, :],
        torch.cat((x, y), axis=0)[None, :, :],
        **kernel_kwargs,
    )
    K_xx = K_xy_xy[:n, :n]
    K_xy = K_xy_xy[:n, n:]

    hx = (
        1 / m * K_xy.sum(axis=1) - K_xx @ torch.from_numpy(alpha).float()
    ) / lambda_

    D = 1 / n * torch.diag(torch.cat((torch.exp(hx), torch.zeros(m))))

    v = torch.cat((1 / n * torch.exp(hx), -1 / m * torch.ones(m)))

    coefs = torch.inverse(D @ K_xy_xy + lambda_ * torch.eye(n + m)) @ v

    if len(eval_pts.shape) == 2:
        k_z_xy = kernel(
            eval_pts[:, None, :],
            torch.cat((x, y), axis=0)[None, :, :],
            **kernel_kwargs,
        )
        hsr = h_star(x, y, eval_pts, alpha, lambda_, kernel, kernel_kwargs)

        ret = ((k_z_xy @ coefs) * hsr.exp()).sum()
    elif len(eval_pts.shape) == 3:
        k_z_xy = kernel(
            eval_pts[:, :, None, :],
            torch.cat((x, y), axis=0)[None, None, :, :],
            **kernel_kwargs,
        )
        hsr = h_star(x, y, eval_pts, alpha, lambda_, kernel, kernel_kwargs)
        ret = ((k_z_xy @ coefs) * hsr.exp()).sum() / eval_pts.shape[0]
    else:
        raise ValueError("eval_pts should have 2 or 3 dimensions")

    return penalized_grad + (1 + lambda_) * ret


def h_star(x, y, eval_pts, alpha, lambda_, kernel, kernel_kwargs):
    # TODO: refactor the code to use this more!
    if len(eval_pts.shape) == 2:
        kzx = kernel(eval_pts[:, None, :], x[None, :, :], **kernel_kwargs)
        kzy = kernel(eval_pts[:, None, :], y[None, :, :], **kernel_kwargs)

        ret1 = 1 / y.shape[0] * kzy.sum(axis=-1)
        if isinstance(kzx, torch.Tensor):
            ret2 = kzx @ torch.from_numpy(alpha).float()
        else:
            ret2 = kzx @ alpha
        return (ret1 - ret2) / lambda_
    elif len(eval_pts.shape) == 3:
        kzy = kernel(
            eval_pts[:, :, None, :], y[None, None, :, :], **kernel_kwargs
        )
        kzx = kernel(
            eval_pts[:, :, None, :], x[None, None, :, :], **kernel_kwargs
        )
        ret1 = 1 / y.shape[0] * kzy.sum(axis=-1)

        if isinstance(kzx, torch.Tensor):
            ret2 = torch.matmul(kzx, torch.from_numpy(alpha).float())
        else:
            ret2 = kzx @ alpha
        w = ret1 - ret2
        return w / lambda_
    else:
        raise ValueError


def get_dual_gap(alpha, K_xx, K_xy, K_yy, lambda_):
    primal = _kale_primal(
        alpha,
        K_xx,
        K_xy,
        K_yy,
        lambda_,
        {},
        penalized=True,
        also_return_norm_term=False,
    )
    dual = _kale_dual(alpha, K_xx, K_xy, K_yy, lambda_, {}, penalized=True)
    return primal - dual


def kale_coordinate_descent(
    alpha_0, lambda_, max_iter, dual_gap_tol, K_xx, K_xy, K_yy
):
    # tol = 1e-10
    logger = get_logger("kale.optim.cd")

    assert K_xx is not None
    assert K_xy is not None
    # assert K_xy.shape[0] == K_xy.shape[1]
    N, M = K_xy.shape
    one_n_KXY_t_1 = 1 / K_xy.shape[1] * K_xy @ np.ones(K_xy.shape[1])
    alpha = alpha_0.copy()

    for j in range(max_iter):
        for i in range(len(alpha)):
            _v = (
                one_n_KXY_t_1[i] - (K_xx[i, :] @ alpha - K_xx[i, i] * alpha[i])
            ) / lambda_
            high_lambda = _v > 10
            low_lambda = _v < -50
            if high_lambda:
                # Use asymptotic development of lambertw in +\infty
                # prevents overflow of np.exp
                alpha[i] = (
                    lambda_
                    / K_xx[i, i]
                    * (
                        np.log(K_xx[i, i] / (lambda_ * N))
                        + _v
                        - np.log(np.log(K_xx[i, i] / (lambda_ * N)) + _v)
                    )
                )
            elif low_lambda:
                # Use taylor series development of lambertw in 0
                # prevents convergence errors of lambertw
                alpha[i] = (
                    lambda_
                    / K_xx[i, i]
                    * (
                        K_xx[i, i] * np.exp(_v) / (lambda_ * N)
                        - (K_xx[i, i] * np.exp(_v) / (lambda_ * N)) ** 2
                    )
                )
            else:
                alpha[i] = (
                    lambda_
                    / K_xx[i, i]
                    * lambertw(K_xx[i, i] * np.exp(_v) / (lambda_ * N)).real
                )

        # TODO (pierreglaser): micro-optimization: kale will re-compute the
        # primal and dual value - while it is already computed here.
        dual_gap = get_dual_gap(alpha, K_xx, K_xy, K_yy, lambda_)
        if dual_gap < dual_gap_tol:
            break
        logger.info(
            "iter {j}, dual gap: {dual_gap:.3f}".format(j=j, dual_gap=dual_gap)
        )
    else:
        logger.warning(
            "convergence was not reached after {} iterations (dual gap: {}, "
            "objective dual gap: {})".format(j, dual_gap, dual_gap_tol)
        )

    kale_estimation_info = {}
    return alpha, kale_estimation_info


def online_kale_coordinate_descent(
    alpha_0,
    lambda_,
    max_iter,
    dual_gap_tol,
    X,
    Y,
    kernel,
    kernel_kwargs,
    buffer_size=1000,
):
    # Efficient, online computation of KALE. Code is super complicated because
    # I compute kale on the fly, while doing the coordinate updates. This
    # allows me to check the dual gap at each iteration without having to
    # reloop through all the data, which would double the runtime cost.

    logger = get_logger("kale.optim.online_cd")
    # same parameter as in _kale_dual
    _log_tol = 1e-50

    assert X is not None
    assert Y is not None

    assert kernel is not None
    assert kernel_kwargs is not None

    alpha = alpha_0.copy()

    Nx = N = len(alpha)
    Ny = M = len(Y)

    all_idxs = np.arange(N)

    if (N % buffer_size) == 0:
        chunks = [
            all_idxs[(i * buffer_size) : (i + 1) * buffer_size]  # noqa
            for i in range(N // buffer_size)
        ]
        n_chunks = N // buffer_size
    else:
        chunks = [
            all_idxs[(i * buffer_size) : (i + 1) * buffer_size]  # noqa
            for i in range(1 + (N // buffer_size))
        ]
        n_chunks = N // buffer_size + 1

    # sum(np.log(N * alpha_i)), chunked
    alpha_slice_chunks = np.zeros((n_chunks,))

    # Each 3 term of the norm square (alpha.T @ K_xx @ alpha - 2/Ny * alpha.T @
    # K_xy @ 1 needs a separate computation logic. The quadratic term
    # (alphaKXXalpha) term must be tracked in a 2-d (n_chunks x n_chunks) array
    # to update all terms affected by by one coordinate update. The linear term
    # can be tracked in a simple array of size n_chunk, updating the entry
    # related to each updated coordinate at a time. The constant term can be
    # computed using a single accumulator.

    # quadratic term tracking data structure
    _xx_chunks = np.zeros((n_chunks, n_chunks))

    # linear term tracking data structure
    linear_norm_part = np.zeros((n_chunks,))

    # constant term tracking data structure
    K_yy_sum = 0

    # kale primal terms, chunked
    primal_exp_term = np.zeros((N,))
    primal_nonexp_term_chunks = np.zeros((n_chunks,))

    K_xx_buffer = None
    K_yy_buffer = None
    K_xy_buffer = None

    for j in range(max_iter):
        for chunk_id, chunk in enumerate(chunks):
            K_xx_buffer = kernel(
                X[chunk, None, :], X[None, :, :], **kernel_kwargs
            )
            K_xy_buffer = kernel(
                X[chunk, None, :], Y[None, :, :], **kernel_kwargs
            )
            if j == 0 and chunk[0] < M:
                if chunk[-1] >= M:
                    _y_chunk = chunk[chunk < M]
                else:
                    _y_chunk = chunk
                # necessary to compute kale (not the coordinate updates)
                K_yy_buffer = kernel(
                    Y[_y_chunk, None, :], Y[None, :, :], **kernel_kwargs
                )
                K_yy_sum += K_yy_buffer.sum()

            one_n_KXY_t_1_buffer = 1 / M * K_xy_buffer @ np.ones(M)

            # update the coefficients
            for rel_idx, abs_idx in enumerate(chunk):
                _v = (
                    one_n_KXY_t_1_buffer[rel_idx]
                    - (
                        K_xx_buffer[rel_idx, :] @ alpha
                        - K_xx_buffer[rel_idx, abs_idx] * alpha[abs_idx]
                    )
                ) / lambda_
                high_lambda = _v > 10
                low_lambda = _v < -50
                if high_lambda:
                    # Use asymptotic development of lambertw in +\infty
                    # prevents overflow of np.exp
                    # lambertw = log(x) - log(log(x))
                    _logx = (
                        np.log(K_xx_buffer[rel_idx, abs_idx] / (lambda_ * N))
                        + _v
                    )
                    _loglogx = np.log(_logx)
                    alpha[abs_idx] = (
                        lambda_
                        / K_xx_buffer[rel_idx, abs_idx]
                        * (_logx - _loglogx)
                    )
                elif low_lambda:
                    # Use taylor series development of lambertw in 0
                    # prevents convergence errors of lambertw
                    alpha[abs_idx] = (
                        lambda_
                        / K_xx_buffer[rel_idx, abs_idx]
                        * (
                            K_xx_buffer[rel_idx, abs_idx]
                            * np.exp(_v)
                            / (lambda_ * N)
                            - (
                                K_xx_buffer[rel_idx, abs_idx]
                                * np.exp(_v)
                                / (lambda_ * N)
                            )
                            ** 2
                        )
                    )
                else:
                    alpha[abs_idx] = (
                        lambda_
                        / K_xx_buffer[rel_idx, abs_idx]
                        * lambertw(
                            K_xx_buffer[rel_idx, abs_idx]
                            * np.exp(_v)
                            / (lambda_ * N)
                        ).real
                    )
            # compute KALE dual online
            alpha_slice = alpha[chunk]
            alpha_slice_chunk = np.sum(
                alpha_slice * np.log(_log_tol + Nx * alpha_slice)
            ) - np.sum(alpha_slice)
            alpha_slice_chunks[chunk_id] = alpha_slice_chunk

            xy_part = (K_xy_buffer.T @ alpha_slice).sum() / Ny

            # the alpha.T * K_xx * alpha contains product of alpha chunks, and
            # update one chunk must result in an update of all the produc
            # containing this chunk. So this term must be tracked in the form a
            # a matrix of shape (n_chunks x n_chunks).  this term is used both
            # for the primal and the dual
            _xx_part_unfinished = (alpha_slice @ K_xx_buffer) * alpha
            for _i, _c in enumerate(chunks):
                _xx_chunks[_i, chunk_id] = np.sum(_xx_part_unfinished[_c])
                _xx_chunks[chunk_id, _i] = np.sum(_xx_part_unfinished[_c])

            lnp = 1 / (2 * lambda_) * (-2 * xy_part)
            linear_norm_part[chunk_id] = lnp

            # compute KALE primal online:
            primal_exp_term += -1 / lambda_ * (alpha_slice @ K_xx_buffer)
            primal_exp_term[chunk] += 1 / lambda_ * one_n_KXY_t_1_buffer

            # the missing term of int h dP contains the Kyy sums, which is
            # computed inside a separate accumulator
            _p2 = -(-1 / lambda_ * (alpha_slice @ K_xy_buffer).sum()) / Ny
            primal_nonexp_term_chunks[chunk_id] = _p2

        # finish computing the K_yy sum if needed:
        if j == 0 and M > N:
            y_chunk = N
            while y_chunk < M:
                K_yy_sum += kernel(
                    Y[y_chunk : y_chunk + buffer_size, None, :],  # noqa
                    Y[None, :, :],
                    **kernel_kwargs,
                ).sum()
                y_chunk += buffer_size

        norm_squared_term = (
            np.sum(linear_norm_part)
            + 1 / (2 * lambda_) * np.sum(_xx_chunks)
            + 1 / (2 * lambda_) * (K_yy_sum / (Ny ** 2))
        )

        _kale_dual_val = -(np.sum(alpha_slice_chunks) + norm_squared_term)
        _kale_primal_val = (
            np.exp(primal_exp_term).sum() / Nx
            + np.sum(primal_nonexp_term_chunks)
            + (-1 / lambda_ * (1 / Ny * K_yy_sum) / Ny)
            + norm_squared_term
        )
        dual_gap = _kale_primal_val - _kale_dual_val
        logger.info(
            "iter {j}, dual_gap {dual_gap}".format(j=j, dual_gap=dual_gap)
        )
        if dual_gap < dual_gap_tol:
            break

        # discard some tracking values from previous iteration
        # will NOT WORK if the the CD stops between single coordinate updates!
        if j < (max_iter - 1):
            primal_exp_term[:] = 0
    else:
        logger.warning(
            "convergence was not reached after {} iterations (dual gap: {}, "
            "objective dual gap: {})".format(j, dual_gap, dual_gap_tol)
        )

    kale_estimation_info = {}
    ret = (
        alpha,
        _kale_dual_val,
        _kale_primal_val,
        norm_squared_term,
        kale_estimation_info,
    )

    return ret
