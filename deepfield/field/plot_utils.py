"""Plot utils."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # pylint: disable=unused-import
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from ipywidgets import interact, widgets

COLORS = ['r', 'b', 'm', 'g']


def show_slice_static(component, att, i=None, j=None, k=None, t=None, figsize=None, ax=None, **kwargs):
    """Plot slice of the 3d/4d data array.

    Parameters
    ----------
    component : BaseComponent
        Component containing attribute to show.
    att : str
        Attribute to show.
    i : int or None
        Slice along x-axis to show.
    j : int or None
        Slice along y-axis to show.
    k : int or None
        Slice along z-axis to show.
    t : int or None
        Slice along t-axis to show.
    figsize : array-like, optional
        Output plot size. Ignored if `ax` is provided.
    ax : matplotlib.axes.Axes, optional
        Axes to plot slice. Default is 'auto'.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Plot of a cube slice.
    """
    count = np.sum([i is not None for i in [i, j, k, t]])
    grid = component.field.grid
    data = getattr(component, att)
    if data.ndim == 4:
        if count != 2:
            raise ValueError('Two slices are expected for spatio-temporal data, found {}.'.format(count))
        if t is None:
            raise ValueError('`t` should be provided for spatio-temporal data.')
    elif data.ndim == 3:
        if count != 1:
            raise ValueError('Single slice is expected for spatial data, found {}.'.format(count))
        if t is not None:
            raise ValueError('`t` should not be provided for spatial only data.')
    else:
        raise ValueError('Data should have 3 or 4 dimensions, found {}.'.format(data.ndim))

    dims = 4
    if data.ndim == 3:
        dims = 3
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if dims == 4:
        data = data[t]
    if i is not None:
        points = grid.xyz[i, :, :, ::2, 1:][grid.actnum[i, :, :]]
        n_blocks = grid.actnum[i, :, :].sum()
        colors = np.tile(data[i, :, :][grid.actnum[i, :, :]].reshape(-1,1), (1, 2)).ravel()
        xlabel = 'y'
        ylabel = 'z'
        invert_y = True
    elif j is not None:
        points = grid.xyz[:, j, :,][...,(0, 1, 4, 5), :][..., (0,2)][grid.actnum[:, j, :]]
        n_blocks = grid.actnum[:, j, :].sum()
        colors = np.tile(data[:, j, :][grid.actnum[:, j, :]].reshape(-1,1), (1, 2)).ravel()
        xlabel = 'x'
        ylabel = 'z'
        invert_y = True
    elif k is not None:
        points = grid.xyz[:, :, k, :4, :2][grid.actnum[:, :, k]]
        n_blocks = grid.actnum[:, :, k].sum()
        colors = np.tile(data[:, :, k][grid.actnum[:, :, k]].reshape(-1,1), (1, 2)).ravel()
        xlabel = 'x'
        ylabel = 'y'
        invert_y = False
    else:
        raise ValueError('One of i, j, or k slices should be defined.')

    if n_blocks > 0:
        x, y = points[:, :, 0].ravel(), points[:, :, 1].ravel()
        triangles = np.tile(np.hstack((np.arange(3), np.array([1,2,3]))), (n_blocks, 1))
        triangles = triangles + np.arange(0, n_blocks*4,4).reshape(-1,1)
        triangles = triangles.reshape(-1, 3)
        ax.tripcolor(x, y, colors, triangles=triangles, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if invert_y:
        ax.invert_yaxis()


def show_slice_interactive(component, att, figsize=None, **kwargs):
    """Plot cube slices with interactive sliders.

    Parameters
    ----------
    component : BaseComponent
        Component containing attribute to show.
    att : str
        Attribute to show.
    figsize : array-like, optional
        Output plot size.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Plot of 3 cube slices with interactive sliders.
    """
    if 'origin' in kwargs:
        kwargs = kwargs.copy()
        del kwargs['origin']
    data = getattr(component, att)
    def update(t=None, i=0, j=0, k=0):
        _, axes = plt.subplots(1, 3, figsize=figsize)
        show_slice_static(component, att, i=i, t=t, ax=axes[0])
        show_slice_static(component, att, j=j, t=t, ax=axes[1])
        show_slice_static(component, att, k=k, t=t, ax=axes[2])

    shape = data.shape

    if data.ndim == 3:
        interact(lambda i, j, k: update(None, i, j, k),
                 i=widgets.IntSlider(value=shape[0] / 2, min=0, max=shape[0] - 1, step=1),
                 j=widgets.IntSlider(value=shape[1] / 2, min=0, max=shape[1] - 1, step=1),
                 k=widgets.IntSlider(value=shape[2] / 2, min=0, max=shape[2] - 1, step=1))
    elif data.ndim == 4:
        interact(update,
                 t=widgets.IntSlider(value=shape[0] / 2, min=0, max=shape[0] - 1, step=1),
                 i=widgets.IntSlider(value=shape[1] / 2, min=0, max=shape[1] - 1, step=1),
                 j=widgets.IntSlider(value=shape[2] / 2, min=0, max=shape[2] - 1, step=1),
                 k=widgets.IntSlider(value=shape[3] / 2, min=0, max=shape[3] - 1, step=1))
    else:
        raise ValueError('Invalid data shape. Expected 3 or 4, got {}.'.format(data.ndim))
    plt.show()


def plot_bounds_3d(top, bottom, figsize=None):
    """Plot top and bottom surfaces."""
    x = np.arange(top.shape[0])
    y = np.arange(top.shape[1])
    x, y = np.meshgrid(x, y)
    nans = np.isnan(top)

    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, bottom.T, color='b', edgecolor='none')
    ax.plot_surface(x, y, top.T, color='r', edgecolor='none')
    ax.set_zlim(bottom[~nans].max(), top[~nans].min())
    plt.show()


def plot_bounds_2d(top, bottom, x=None, y=None, figsize=None):
    """Plot top and bottom lines in x and y projections."""
    nans = np.isnan(top)
    ylim = (bottom[~nans].max(), top[~nans].min())

    def update(x=0, y=0):
        _, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(top[x], label='top')
        axes[0].plot(bottom[x], label='bottom')
        axes[0].axvline(y, c='r')
        axes[0].set_xlim((0, top.shape[1]))
        axes[0].set_title('x')

        axes[1].plot(top[:, y], label='top')
        axes[1].plot(bottom[:, y], label='bottom')
        axes[1].axvline(x, c='r')
        axes[1].set_xlim((0, top.shape[0]))
        axes[1].set_title('y')

        for ax in axes:
            ax.legend()
            ax.set_ylim(*ylim)

    if (x is None) and (y is None):
        interact(update,
                 x=widgets.IntSlider(0, 0, top.shape[0] - 1, step=1),
                 y=widgets.IntSlider(0, 0, top.shape[1] - 1, step=1))
    else:
        x = 0 if x is None else x
        y = 0 if y is None else y
        update(x, y)
    plt.show()

def make_patch_spines_invisible(ax):
    """Make patch spines invisible"""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_table_1d(table, figsize=None):
    """
    Plot table with 1-dimensional domain
    Parameters
    ----------
    table: geology.src.tables.tables._Table
        Table to be plotted
    figsize: tuple
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.suptitle(table.name)
    ax.set_xlabel(table.domain[0])
    ax = [ax, ]
    ax_position = 0.8
    for _ in range(len(table.columns) - 1):
        ax_position += 0.2
        ax.append(ax[0].twinx())
        ax[-1].spines["right"].set_position(("axes", ax_position))
        make_patch_spines_invisible(ax[-1])
        ax[-1].spines["right"].set_visible(True)

    x = table.index.values
    for i, col in enumerate(table.columns):
        ax[i].plot(x, table[col].values, color=COLORS[i])
        ax[i].set_ylabel(col, color=COLORS[i])
        ax[i].tick_params(axis='y', labelcolor=COLORS[i])
    plt.show()


def plot_table_2d(table, figsize=None):
    """
    Plot table with 2-dimensional domain
    Parameters
    ----------
    table: geology.src.tables.tables._Table
        Table to be plotted
    figsize: tuple
    """
    domain_names = list(table.domain)
    domain0_value_widget = widgets.SelectionSlider(
        description=domain_names[0],
        options=list(sorted(set(table.index.get_level_values(0))))
    )

    def update(domain0_value):
        cropped_table = table.loc[table.index.get_level_values(0) == domain0_value]
        cropped_table = cropped_table.droplevel(0)
        cropped_table.domain = [domain_names[1]]
        plot_table_1d(cropped_table, figsize)
    interact(update, domain0_value=domain0_value_widget)
