import torch
import numpy as np


def _create_potential_from_gaussian_mixture(mus, sigmas):
    first_sigma = sigmas[0]
    assert isinstance(first_sigma, (float, int))

    def potential_func(x):
        ret = - ((x[:, :, None] - mus[None, :, :]) ** 2).sum(axis=1) / (2 * first_sigma ** 2)  # noqa

        # log-sum-exp trick
        maxs = ret.max(axis=1, keepdims=True).values
        potential = - (
            (ret - maxs).exp().sum(axis=1, keepdims=True).log() + maxs
        ).sum()

        # return - ret.exp().sum(axis=1).log().sum()
        return potential
    return potential_func


def generate_two_gaussians(
    N: int,
    d: int,
    m_x: float,
    m_y: float,
    sigma_x: float,
    sigma_y: float,
    random_seed: int,
    requires_grad=("Y",),
    return_potential_functions=False
):
    torch.manual_seed(random_seed)

    X = np.sqrt(sigma_x) * torch.randn(N, d) + m_x
    Y = np.sqrt(sigma_y) * torch.randn(N, d) + m_y

    assert all(i in ("X", "Y") for i in requires_grad)
    X.requires_grad = "X" in requires_grad
    Y.requires_grad = "Y" in requires_grad

    if return_potential_functions:
        assert isinstance(m_x, (float, int))
        mu_x = torch.Tensor([m_x] * d)
        mus_x = mu_x.reshape(d, 1)
        potential = _create_potential_from_gaussian_mixture(mus_x, [1])
        return X, Y, potential
    else:
        return X, Y


def generate_XY(
    N: int,
    d: int,
    dist: float,
    random_seed: int,
    perturbation_level: float,
    requires_grad=("Y",),
    return_potential_functions=False
):
    rs = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)

    X = rs.randn(N, d) / np.sqrt(d)
    # X = rs.uniform(0, 1, size=N*d).reshape(N, d)

    delta = dist * np.ones((N, d)) / np.sqrt(d)
    Y = X + delta
    assert np.allclose(np.linalg.norm((X - Y), axis=-1), dist)

    X = torch.from_numpy(X).float()

    Y = torch.from_numpy(Y).float()
    Y += perturbation_level / np.sqrt(d) * torch.randn(*Y.shape)

    assert all(i in ("X", "Y") for i in requires_grad)
    X.requires_grad = "X" in requires_grad
    Y.requires_grad = "Y" in requires_grad

    return X, Y


def generate_XY_perturbed_gaussian(
    N: int,
    d: int,
    dist: float,
    random_seed: int,
    perturbation_level: float,
    requires_grad=("Y",),
    return_potential_functions=False
):
    rs = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)

    std = 2
    scale = 1/100

    Y = rs.randn(N, d) / np.sqrt(d)
    Y /= std
    # X = rs.uniform(0, 1, size=N*d).reshape(N, d)

    X = np.zeros((N, d))
    clusters = np.array([-2, -1, 1, 2])
    cluster_algrebraic_idx = rs.choice(clusters, size=N).reshape(N, 1)

    delta = cluster_algrebraic_idx * dist * np.ones((N, d)) / np.sqrt(d)
    X = Y + delta
    # assert np.allclose(np.linalg.norm((X - Y), axis=-1), dist)

    X = torch.from_numpy(X).float()

    Y = torch.from_numpy(Y).float()
    Y += perturbation_level / np.sqrt(d) * torch.randn(*Y.shape)

    X *= scale
    Y *= scale

    assert all(i in ("X", "Y") for i in requires_grad)
    X.requires_grad = "X" in requires_grad
    Y.requires_grad = "Y" in requires_grad

    return X, Y


def generate_XY_mog(
    N: int,
    d: int,
    dist: float,
    random_seed: int,
    perturbation_level: float,
    requires_grad=("Y",),
    return_potential_functions=False
):
    rs = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)

    std = 2
    # scale = 1/100
    scale = 1

    Y = rs.randn(N, d) / np.sqrt(d)
    Y /= std
    # X = rs.uniform(0, 1, size=N*d).reshape(N, d)

    X = np.zeros((N, d))

    direction_one = np.ones((N, d))
    direction_two = np.ones((N, d)) * [[1] + [-1] + [1] * (d - 2)]

    clusters_level_1 = np.array([-2, -1, 0, 1, 2])
    clusters_level_0 = np.array([0, 1])

    cluster_algrebraic_idx_lvl1 = rs.choice(clusters_level_1, size=N).reshape(N, 1)
    cluster_algrebraic_idx_lvl0 = rs.choice(clusters_level_0, size=N).reshape(N, 1)

    delta = cluster_algrebraic_idx_lvl0 * cluster_algrebraic_idx_lvl1 * dist * direction_one / np.sqrt(d)
    delta += (1 - cluster_algrebraic_idx_lvl0) * cluster_algrebraic_idx_lvl1 * dist * direction_two / np.sqrt(d)
    X = Y + delta
    Y *= 6
    # assert np.allclose(np.linalg.norm((X - Y), axis=-1), dist)

    X = torch.from_numpy(X).float()

    Y = torch.from_numpy(Y).float()
    Y += perturbation_level / np.sqrt(d) * torch.randn(*Y.shape)

    X *= scale
    Y *= scale

    assert all(i in ("X", "Y") for i in requires_grad)
    X.requires_grad = "X" in requires_grad
    Y.requires_grad = "Y" in requires_grad

    return X, Y


def generate_mog_square(
    N: int,
    d: int,
    dist: float,
    std: float,
    random_seed: int,
    requires_grad: bool,
    return_potential_functions=False
):
    # generate isotropic gaussians at the 4 corners of a square
    assert d == 2

    rs = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)

    ms = torch.Tensor([[0, dist], [dist, 0], [dist, dist], [0, 0]]).T
    num_clusters = ms.shape[1]

    idxs = rs.choice(num_clusters, N)
    means = np.array([ms.numpy()[:, idx] for idx in idxs])
    X = std * torch.randn(N, 2, requires_grad=False) + means
    X.requires_grad = requires_grad

    if return_potential_functions:
        sigmas = [std] * num_clusters
        potential = _create_potential_from_gaussian_mixture(ms, sigmas)
        return X, potential
    else:
        return X


def generate_XY_mog_square(
    N: int,
    d: int,
    dist: float,
    std: float,
    random_seed: int,
    requires_grad=("Y",),
    y_rel_dist=1.5,
    y_std=1,
    return_potential_functions=False
):
    assert all(i in ("X", "Y") for i in requires_grad)
    X_requires_grad = "X" in requires_grad
    Y_requires_grad = "Y" in requires_grad

    _x_generator_args = (N, d, dist, std, random_seed, X_requires_grad)
    if return_potential_functions:
        X, potential = generate_mog_square(
            *_x_generator_args, return_potential_functions=True
        )
    else:
        X = generate_mog_square(
            *_x_generator_args, return_potential_functions=False
        )

    m_y = - y_rel_dist * dist
    Y = torch.randn(N, d, requires_grad=Y_requires_grad)

    with torch.no_grad():
        Y *= y_std
        Y += m_y

    if return_potential_functions:
        return X, Y, potential
    else:
        return X, Y


def generate_XY_circle(
    N: int, d: int, diameter_ratio: float, noise_level: int, random_seed: int,
    M=None, return_potential_functions=False
):
    rs = np.random.RandomState(random_seed)
    if M is not None:
        max_N_M = max(N, M)
    else:
        M = N
        max_N_M = N

    assert d == 2
    Y = np.c_[
        5 * np.cos(np.linspace(0, 2 * np.pi, max_N_M + 1)),
        5 * np.sin(np.linspace(0, 2 * np.pi, max_N_M + 1)),
    ][
        :-1
    ]  # noqa
    X = Y / diameter_ratio
    Y += (
        noise_level
        * rs.randn(max_N_M, d)
        / np.sqrt(d)
    )
    _N_idxs = rs.choice(max_N_M, size=N, replace=False)
    _M_idxs = rs.choice(max_N_M, size=M, replace=False)

    X = X[_N_idxs]
    Y = Y[_M_idxs]

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    Y.requires_grad = True
    return X, Y


def _get_xy_range(X, trajectories):
    pad_ratio=1.2
    min_x = pad_ratio * min(X[:, 0].min(), trajectories[:, :, 0].min())
    max_x = pad_ratio * max(X[:, 0].max(), trajectories[:, :, 0].max())
    min_y = pad_ratio * min(X[:, 1].min(), trajectories[:, :, 1].min())
    max_y = pad_ratio * max(X[:, 1].max(), trajectories[:, :, 1].max())
    return min_x, max_x, min_y, max_y


def compute_velocity_field(
    X,
    trajectories,
    kernel,
    kernel_kwargs,
    loss,
    loss_kwargs,
    loss_first_variation,
    loss_states,
    xy_lims=None,
):
    assert len(trajectories) == len(loss_states)
    assert len(X.shape) == 2

    if xy_lims is None:
        min_x, max_x, min_y, max_y = _get_xy_range(X, trajectories)
    else:
        min_x, max_x, min_y, max_y = xy_lims


    l_x = torch.linspace(min_x, max_x, steps=50)
    l_y = torch.linspace(min_y, max_y, steps=50)

    velocity_eval_pts = torch.cartesian_prod(l_x, l_y)

    velocity_fields = []

    X = torch.from_numpy(X)

    for Y, info in zip(trajectories, loss_states):

        Y = torch.from_numpy(Y)

        # velocity_eval_pts = torch.from_numpy(Y.detach().numpy())

        velocity_eval_pts.requires_grad = True

        first_val_ = loss_first_variation(
            X, Y, velocity_eval_pts, kernel, kernel_kwargs, info
        )
        first_val_.backward()

        velocity_fields.append(
            (
                velocity_eval_pts.detach().numpy().copy(),
                -velocity_eval_pts.grad.detach().numpy().copy(),
            )
        )

        velocity_eval_pts.grad.data.zero_()
    return velocity_fields
