"""
Plotting helpers for band structures, accessible as ``mnx.io``.

The functions take the output of :meth:`mnx.DynMatrix.get_bands` or
:meth:`mnx.DynMatrix.get_bands_with_SF` (and of :meth:`mnx.Bands.get_bands`) directly::

    bands, qpath, xticks = dyn.get_bands(k_inpath, N=50)
    fig, ax = plt.subplots()
    mnx.io.plot_bands(ax, qpath, bands, xticks, ["$\\Gamma$", "M", "K", "$\\Gamma$"])

All of them draw on a matplotlib ``Axes`` and return it, so several data sets can
be overlaid on the same panel.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm

from mnx.FModules import io as _fio


def plot_bands(ax, qpath, bands, xticks, xlabels, color="tab:blue", lw=1.5, alpha=0.8, label=None, ls="-"):
    """
    Plot a band structure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    qpath : np.ndarray, [Nq]
        Position of each point along the path (``cart_qpath`` from ``get_bands``).
    bands : np.ndarray, [Nq, Nbands]
        Energies of each band, e.g. phonon frequencies in cm^-1. Only the real part is used.
    xticks : list of float
        Path positions of the high-symmetry points.
    xlabels : list of str
        Labels of the high-symmetry points, one per tick.
    color, lw, alpha, ls : optional
        Line colour, width, transparency and style, shared by all bands.
    label : str, optional
        Legend entry for the whole band structure (attached to the first band only).

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``.
    """
    bands = np.real(bands)
    lines = ax.plot(qpath, bands, color=color, alpha=alpha, lw=lw, ls=ls)
    if label is not None:
        lines[0].set_label(label)
    _decorate(ax, qpath, xticks, xlabels, ylim=(bands.min() - 10, bands.max() + 10))
    ax.set_ylabel(r"$\omega$ (cm$^{-1}$)", fontsize=12)
    return ax


def plot_bands_segment(qpath, bands, data, xticks, xlabels, ax=None, vmin=1e-2, vmax=1, cmap="plasma", num_interp=4000):
    """
    Plot a band structure with every band coloured by a weight, e.g. the structure factor.

    The colour (and opacity) of each band at each point follows ``data`` on a
    logarithmic scale between ``vmin`` and ``vmax``; bands with a weight below
    ``vmin`` fade out. The bands are resampled on ``num_interp`` points for a smooth
    colour gradient.

    Parameters
    ----------
    qpath : np.ndarray, [Nq]
        Position of each point along the path.
    bands : np.ndarray, [Nq, Nbands]
        Energies of each band (cm^-1 for phonons).
    data : np.ndarray, [Nq, Nbands]
        Weight of each band at each point, e.g. ``structure_factor`` from
        ``get_bands_with_SF``. Its absolute value is used.
    xticks : list of float
        Path positions of the high-symmetry points.
    xlabels : list of str
        Labels of the high-symmetry points.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    vmin, vmax : float, optional
        Limits of the logarithmic colour scale. Default to 1e-2 and 1.
    cmap : str, optional
        Matplotlib colour map; its opacity is ramped from transparent to opaque.
    num_interp : int, optional
        Number of points used to resample the path. Defaults to 4000.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot and a colour bar.
    """
    if ax is None:
        _, ax = plt.subplots()
    bands = np.real(bands)
    data = np.abs(data)

    # Colour map whose opacity grows quadratically with the weight.
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
    colors[:, -1] = np.linspace(0, 1, 256) ** 2
    alpha_cmap = ListedColormap(colors)
    norm = LogNorm(vmin, vmax)

    ax.plot(qpath, bands, color=plt.get_cmap(cmap)(0.0), alpha=0.1)

    x = np.linspace(qpath[0], qpath[-1], num_interp)
    y = _resample(qpath, bands, x)
    z = _resample(qpath, data, x)
    for band in range(bands.shape[1]):
        points = np.stack([x, y[:, band]], axis=1)[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap=alpha_cmap, norm=norm, linewidth=1, capstyle="round")
        collection.set_array(z[:-1, band])
        ax.add_collection(collection)

    _decorate(ax, qpath, xticks, xlabels, ylim=(bands.min() - 10, bands.max() + 10))
    cbar = ax.figure.colorbar(collection, ax=ax)
    cbar.set_label("Dynamical structure-factor (a.u.)", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    cbar.solids.set(alpha=None)
    return ax


def map_from_bands(ax, qpath, bands, xticks, xlabels, data, emin, emax, Ny=1000, Nx_uniform=500, sigma=10, vmin=0, vmax=None, dy=20, cmap="plasma", bcolor="black"):
    r"""
    Plot a weighted band structure as a broadened intensity map.

    Each band contributes a Gaussian of width ``sigma`` around its energy, scaled by
    its weight:

    .. math:: I(x, E) = \sum_\nu w_\nu(x) \exp\left[-\frac{(E - \omega_\nu(x))^2}{2\sigma^2}\right]

    The map is computed by the Fortran routine ``FModules.io.get_map_from_bands`` on
    the points of ``qpath`` and resampled on ``Nx_uniform`` evenly spaced points. The
    bands are drawn on top with :func:`plot_bands`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    qpath : np.ndarray, [Nq]
        Position of each point along the path.
    bands : np.ndarray, [Nq, Nbands]
        Energies of each band (cm^-1 for phonons).
    xticks : list of float
        Path positions of the high-symmetry points.
    xlabels : list of str
        Labels of the high-symmetry points.
    data : np.ndarray, [Nq, Nbands]
        Weight of each band at each point, e.g. ``structure_factor``.
    emin, emax : float
        Energy window shown.
    Ny : int, optional
        Number of energy points of the map. Defaults to 1000.
    Nx_uniform : int, optional
        Number of evenly spaced path points of the map. Defaults to 500.
    sigma : float, optional
        Gaussian broadening, in the units of ``bands``. Defaults to 10.
    vmin, vmax : float, optional
        Colour scale limits. ``vmax`` defaults to the largest weight.
    dy : float, optional
        Extra energy margin computed beyond ``[emin, emax]``. Defaults to 20.
    cmap : str, optional
        Matplotlib colour map of the intensity. Defaults to ``"plasma"``.
    bcolor : str, optional
        Colour of the bands drawn on top. Defaults to ``"black"``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same ``ax``.
    mmap : np.ndarray, [Ny, Nx_uniform]
        The intensity map, energy along the first axis.
    grid_x : np.ndarray, [Nx_uniform]
        Path positions of the map columns.
    grid_y : np.ndarray, [Ny]
        Energies of the map rows.
    """
    if vmax is None:
        vmax = np.real(data.max())

    grid_y = np.linspace(emin - dy, emax + dy, Ny)
    mmap = _fio.get_map_from_bands(qpath, grid_y, bands, data, sigma)  # [len(qpath), Ny]
    grid_x = np.linspace(qpath[0], qpath[-1], Nx_uniform)
    mmap = _resample(qpath, mmap, grid_x).T

    ax.imshow(mmap, interpolation="bilinear", origin="lower", aspect="auto",
              extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], vmin=vmin, vmax=vmax, cmap=cmap)
    plot_bands(ax, qpath, bands, xticks, xlabels, color=bcolor, alpha=0.2)
    ax.set_ylim(emin, emax)
    return ax, mmap, grid_x, grid_y


def gaussian(freq, sigma):
    """Normalized Gaussian of width ``sigma`` evaluated at ``freq``."""
    return 1 / (sigma * np.sqrt(np.pi * 2)) * np.exp(-0.5 * (np.conj(freq) * freq) / sigma**2)


def _decorate(ax, qpath, xticks, xlabels, ylim):
    """Shared styling of band plots: limits, zero line, high-symmetry ticks and lines."""
    ax.set_ylim(*ylim)
    ax.set_xlim(0, qpath[-1])
    ax.tick_params(labelsize=12)
    ax.axhline(0, color="grey", linestyle="dashed", lw=0.5)
    for x in xticks:
        ax.axvline(x, color="grey", linestyle="dashed", lw=0.5)
    ax.set_xticks(xticks, labels=xlabels)


def _resample(x, y, x_new):
    """
    Linear interpolation of ``y`` [len(x), ...] at ``x_new`` along the first axis.

    ``x`` must be non-decreasing; repeated values (the segment joints of a band path)
    are allowed.
    """
    x = np.asarray(x)
    index = np.clip(np.searchsorted(x, x_new, side="right") - 1, 0, len(x) - 2)
    t = (x_new - x[index]) / (x[index + 1] - x[index])
    t = t.reshape(t.shape + (1,) * (np.ndim(y) - 1))
    return (1 - t) * y[index] + t * y[index + 1]
