"""Plot utils."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # pylint: disable=unused-import
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from ipywidgets import interact, widgets

COLORS = ['r', 'b', 'm', 'g']

def get_slice_trisurf(component, att, i=None, j=None, k=None, t=None):
    """Get slice surface triangulaution for further plotting

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
    Returns
    -------
    np.ndarray or None, np.ndarray or None, np.ndarray or None, np.ndarray or None,
    np.ndarray or None
        x-coordinates of vertices, y-coordinate of vertices, triangles, data,
        cell indices corresponding to triangles
    """
    count = np.sum([i is not None for i in [i, j, k, t]])
    grid = component.field.grid
    xyz = grid.xyz
    actnum = grid.actnum
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
    if dims == 4:
        data = data[t]

    if i is not None:
        points = xyz[i, :, :, ::2, 1:][actnum[i, :, :]]
        n_blocks = actnum[i, :, :].sum()
        data = np.tile(data[i, :, :][actnum[i, :, :]].reshape(-1,1), (1, 2)).ravel()
        indices = np.indices(grid.dimens)[:, i, :, :][...,actnum[i, :, :]]
    elif j is not None:
        points = xyz[:, j, :,][...,(0, 1, 4, 5), :][..., (0,2)][actnum[:, j, :]]
        n_blocks = actnum[:, j, :].sum()
        data = np.tile(data[:, j, :][actnum[:, j, :]].reshape(-1,1), (1, 2)).ravel()
        indices = np.indices(grid.dimens)[:, :, j, :][..., actnum[:, j, :]]
    elif k is not None:
        points = xyz[:, :, k, :4, :2][actnum[:, :, k]]
        n_blocks = actnum[:, :, k].sum()
        data = np.tile(data[:, :, k][actnum[:, :, k]].reshape(-1,1), (1, 2)).ravel()
        indices = np.indices(grid.dimens)[:, :, :, k][..., actnum[:, :, k]]
    else:
        raise ValueError('One of i, j, or k slices should be defined.')

    if n_blocks > 0:
        x, y = points[:, :, 0].ravel(), points[:, :, 1].ravel()
        triangles = np.tile(np.hstack((np.arange(3), np.array([1,2,3]))), (n_blocks, 1))
        triangles = triangles + np.arange(0, n_blocks*4,4).reshape(-1,1)
        triangles = triangles.reshape(-1, 3)
        indices = np.tile(indices[..., np.newaxis], (1, 1, 2)).reshape(3, -1).T
        return x, y, triangles, data, indices
    return None, None, None, None, None

def show_slice_static(component, att, i=None, j=None, k=None, t=None,
                      i_line=None, j_line=None, k_line=None,
                      figsize=None, ax=None, **kwargs):
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
    i_line: int, optional
        Plot line corresponding to specific i index.
    j_line: int, optional
        Plot line corresponding to specific j index.
    k_line: int, optional
        Plot line corresponding to specific j index.
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
    grid = component.field.grid
    xyz = grid.xyz
    actnum = grid.actnum


    lines = []
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if i is not None:
        xlabel = 'y'
        ylabel = 'z'
        invert_y = True
        if j_line is not None:
            lines.append(xyz[i, j_line, :,][..., (0, 4), 1:][actnum[i, j_line, :]].reshape(-1, 2))
        if k_line is not None:
            lines.append(xyz[i,:, k_line][..., (0, 2), 1:][actnum[i, :, k_line]].reshape(-1, 2))
        if i_line is not None:
            raise ValueError('`i_line` should be None for i-slice')
    elif j is not None:
        xlabel = 'x'
        ylabel = 'z'
        invert_y = True
        if i_line is not None:
            lines.append(xyz[i_line, j, :,][..., (0, 4), ::2][actnum[i_line, j, :]].reshape(-1, 2))
        if k_line is not None:
            lines.append(xyz[:, j, k_line][..., (0, 1), ::2][actnum[:, j, k_line]].reshape(-1, 2))
        if j_line is not None:
            raise ValueError('`j_line` should be None for j-slice')
    elif k is not None:
        xlabel = 'x'
        ylabel = 'y'
        invert_y = False
        if i_line is not None:
            lines.append(xyz[i_line, :, k, :4:2, :2][actnum[i_line, :, k]].reshape(-1, 2))
        if j_line is not None:
            lines.append(xyz[:, j_line, k, :2, :2][actnum[:, j_line, k]].reshape(-1, 2))
        if k_line is not None:
            raise ValueError('`k_line` should be None for i-slice')
    else:
        raise ValueError('One of i, j, or k slices should be defined.')

    x, y, triangles, colors, _ = get_slice_trisurf(component, att, i, j, k, t)
    if triangles is not None:
        ax.tripcolor(x, y, colors, triangles=triangles, **kwargs)
        for line in lines:
            x, y = line[:, 0], line[:, 1]
            ax.plot(x, y, color='red')

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
        axes = []
        fig = plt.figure(figsize=figsize)
        axes.append(fig.add_subplot(2, 2, 3))
        axes.append(fig.add_subplot(2, 2, 4, sharey=axes[0]))
        axes.append(fig.add_subplot(2, 1, 1))
        show_slice_static(component, att, i=i, t=t, ax=axes[0], j_line=j, k_line=k, **kwargs)
        show_slice_static(component, att, j=j, t=t, ax=axes[1], i_line=i, k_line=k, **kwargs)
        show_slice_static(component, att, k=k, t=t, ax=axes[2], i_line=i, j_line=j, **kwargs)

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
