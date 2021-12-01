import inspect
import pandas as pd
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.gridspec as grispec

from abc import ABC, abstractmethod
from typing import Tuple

from matplotlib.artist import Artist
from matplotlib.collections import Collection

from matplotlib import rc  # noqa
from ipywidgets import interact  # type: ignore

# make it possible to write math formulas using latex inside matplotlib
# rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# rc("text", usetex=True)


class FigAnimation:
    def __init__(self, animations, f, nrows=None, ncols=None):
        self.animations = animations
        self.f = f

        if nrows is not None or ncols is not None:
            assert nrows is not None and ncols is not None
        else:
            nrows = 1
            ncols = len(self.animations)
        self.nrows = nrows
        self.ncols = ncols

    def init(self):
        for i, a in enumerate(self.animations):
            ax = self.f.add_subplot(self.nrows, self.ncols, i + 1)
            a.init(ax)

    def update(self, n):
        for a in self.animations:
            a.update(n)


class AxisAnimation(ABC):
    @abstractmethod
    def init(self, ax):
        pass

    def num_axes(self):
        return 1

    @abstractmethod
    def update(self, n):
        pass


def get_scale(U, V, span=1, N=None):
    # extracted from matplotlib/lib/quiver.py
    import math
    a = np.abs(U + V * 1j)
    if N is None:
        N = len(U)

    sn = max(10, math.sqrt(N))
    amean = a.mean()
    # crude auto-scaling
    # scale is typical arrow length as a multiple of the arrow width
    scale = 1.8 * amean * sn / span
    return scale


def _get_xy_range(X, trajectories):
    pad_ratio = 1.2
    min_x = pad_ratio * min(X[:, 0].min(), trajectories[:, :, 0].min())
    max_x = pad_ratio * max(X[:, 0].max(), trajectories[:, :, 0].max())
    min_y = pad_ratio * min(X[:, 1].min(), trajectories[:, :, 1].min())
    max_y = pad_ratio * max(X[:, 1].max(), trajectories[:, :, 1].max())
    return min_x, max_x, min_y, max_y


class TrajectoryAnimation(AxisAnimation):
    def __init__(
        self,
        X,
        Y,
        trajectories,
        records: pd.DataFrame,
        title,
        metrics_subset,
        velocities=None,
        scale='matplotlib',
        xy_lims=None,
    ):
        self.X = X
        self.Y = Y
        self.trajectories = trajectories
        self.records = records
        self.title = title
        self.metrics_subset = metrics_subset
        self.velocities = velocities
        self._scale = scale
        self._xy_lims = xy_lims

    def init(self, ax):
        self.state = ()
        self.ax = ax
        ax.scatter(self.X[:, 0], self.X[:, 1])

        if self._xy_lims is None:
            min_x, max_x, min_y, max_y = _get_xy_range(self.X, self.trajectories)
        else:
            min_x, max_x, min_y, max_y = self._xy_lims

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        _plot_scatter = ax.scatter(
            self.trajectories[0][:, 0], self.trajectories[0][:, 1]
        )
        self._plot_scatter = _plot_scatter
        return _plot_scatter, ax

    def _get_quiver_scale(self, U, V, span=1):
        if isinstance(self._scale, (int, float)):
            return self._scale
        if self._scale == 'matplotlib':
            # hardcode self.velocities - extreme hack
            # to get the velocities
            return get_scale(U, V, span)
        else:
            raise ValueError

    def update(self, n):
        for artist in self.state:
            artist.remove()

        if self.velocities is not None:
            scale = self._get_quiver_scale(
                self.velocities[n][1][:, 0],
                self.velocities[n][1][:, 1],
                span=1
            )

            q = self.ax.quiver(
                self.velocities[n][0][:, 0],
                self.velocities[n][0][:, 1],
                self.velocities[n][1][:, 0],
                self.velocities[n][1][:, 1],
                scale=scale,
            )
            self.state = (q,)
        self._plot_scatter.set_offsets(self.trajectories[n][:, :2])
        return (self._plot_scatter,)


class StaticPlotAnimation(AxisAnimation):
    def __init__(self, records, metric_name):
        self.records = records
        self.metric_name = metric_name

    def init(self, ax):
        self.ax = ax
        self.records[self.metric_name].plot(ax=self.ax)
        self.ax.set_title(self.metric_name)

    def update(self, n):
        pass


class DynamicLineAnimation(AxisAnimation):
    def __init__(self, x, ys, name):
        self.x = x
        self.ys = ys
        self.name = name

    def init(self, ax):
        self.ax = ax

        (self.l,) = self.ax.plot([], [], label=self.name)

        self.ax.set_xlim(min(*self.x), max(*self.x))

        self.ax.set_ylim(
            max(1e-10, min(*np.array(self.ys).reshape(-1,))),
            max(*np.array(self.ys).reshape(-1,)),
        )

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        self.ax.legend()
        return (self.l,)

    def update(self, n):
        self.l.set_data(self.x, self.ys[n])


def _init_trajectory_plot(
    X,
    Y,
    trajectories,
    records: pd.DataFrame,
    title,
    metrics_subset,
    velocities=None,
    f=None,
):
    # TODO: remove this. use *Animation API instead
    # this function is stateless - no need to rely on closures
    if records is None:
        assert metrics_subset is None or metrics_subset == []
        ax1 = f.add_subplot(1, 1, 1)

    elif len(metrics_subset) > 1:

        num_metrics = len(metrics_subset)
        gs = grispec.GridSpec(num_metrics, 2)
        for i, c in enumerate(metrics_subset):
            ax = f.add_subplot(gs[i, 1])
            ax.set_yscale("log")
            records[[c]].plot(ax=ax)

        ax1 = f.add_subplot(gs[:, 0])
    else:
        # small optimization - GridSpec can take a while to load
        metric_name = metrics_subset[0]
        ax1 = f.add_subplot(1, 2, 1)
        ax2 = f.add_subplot(1, 2, 2)
        records[metric_name].plot(ax=ax2)
        ax2.set_title(title)

    ax1.scatter(X[:, 0], X[:, 1])

    pad_ratio = 1.2

    min_x = pad_ratio * min(X[:, 0].min(), trajectories[:, :, 0].min())
    max_x = pad_ratio * max(X[:, 0].max(), trajectories[:, :, 0].max())
    min_y = pad_ratio * min(X[:, 1].min(), trajectories[:, :, 1].min())
    max_y = pad_ratio * max(X[:, 1].max(), trajectories[:, :, 1].max())

    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)

    _plot_scatter = ax1.scatter(trajectories[0][:, 0], trajectories[0][:, 1])
    return _plot_scatter, ax1


def make_updater(
    _plot_scatter: Collection,
    axis,
    trajectories,
    velocities,
    prev_artists: Tuple[Artist],
):
    # TODO: remove this function, use *Animation API instead
    # This function must keep a state, which is the currently existing artists,
    # that must be removed/updated in-between the iterations - hence we use a
    # closure
    def update(n):
        nonlocal prev_artists
        for artist in prev_artists:
            artist.remove()
        if velocities is not None:
            q = axis.quiver(
                velocities[n][0][:, 0],
                velocities[n][0][:, 1],
                velocities[n][1][:, 0],
                velocities[n][1][:, 1],
            )
            prev_artists = (q,)
        _plot_scatter.set_offsets(trajectories[n][:, :2])

    return update


def vizualize_results(
    X,
    Y,
    trajectories,
    records: pd.DataFrame,
    title,
    metrics_subset,
    velocities=None,
):
    # TODO: use *Animation API inside
    f = plt.figure(figsize=(10, 6), tight_layout=True)

    _plot_scatter, ax1 = _init_trajectory_plot(
        X, Y, trajectories, records, title, metrics_subset, velocities, f
    )

    max_iter = len(trajectories)

    update_fn = make_updater(_plot_scatter, ax1, trajectories, velocities, ())

    # @interact(n=(0, max_iter - 2, max(max_iter // 100, 1)))
    # def f(n):
    #     return update_fn(n)

    interacter = interact(n=(0, max_iter - 2, max(max_iter // 100, 1)))

    update_fn = make_updater(_plot_scatter, ax1, trajectories, velocities, ())
    return interacter(update_fn), ax1
