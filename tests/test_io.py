"""Tests of the plotting helpers in mnx.io with the output of DynMatrix.get_bands*."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import mnx
from mnx.utils.io import _resample

K_INPATH = [[[0, 0, 0], [0.5, 0, 0]], [[0.5, 0, 0], [0, 0, 0]]]
LABELS = [r"$\Gamma$", "M", r"$\Gamma$"]


@pytest.fixture(scope="module")
def bands_sf(crsb):
    return crsb.get_bands_with_SF(K_INPATH, 20, [0, 0, 3])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_bands(crsb):
    bands, qpath, xticks = crsb.get_bands(K_INPATH, 20)
    _, ax = plt.subplots()
    assert mnx.io.plot_bands(ax, qpath, bands, xticks, LABELS, label="mnx") is ax
    band_lines = [line for line in ax.lines if len(line.get_xdata()) == len(qpath)]
    assert len(band_lines) == bands.shape[1]
    np.testing.assert_array_equal(np.array([line.get_ydata() for line in band_lines]).T, bands)
    assert ax.get_legend_handles_labels()[1] == ["mnx"]
    assert [t.get_text() for t in ax.get_xticklabels()] == LABELS


def test_plot_bands_segment_draws_every_band(crsb, bands_sf):
    bands, qpath, xticks, sf = bands_sf
    ax = mnx.io.plot_bands_segment(qpath, bands, sf, xticks, LABELS)
    weighted = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_array() is not None]
    assert len(weighted) == crsb.structure.Natoms * 3


def test_plot_bands_segment_any_number_of_bands():
    qpath = np.linspace(0, 1, 11)
    bands = np.outer(qpath, np.arange(1, 6)) * 100  # 5 bands
    ax = mnx.io.plot_bands_segment(qpath, bands, np.ones_like(bands), [0, 1], ["A", "B"], num_interp=50)
    weighted = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_array() is not None]
    assert len(weighted) == 5


def test_map_from_bands_peaks_on_the_bands():
    qpath = np.linspace(0, 1, 30)
    bands = np.column_stack([np.full(30, 100.0), np.full(30, 300.0)])
    weights = np.column_stack([np.ones(30), np.zeros(30)])
    _, ax = plt.subplots()
    ax, mmap, grid_x, grid_y = mnx.io.map_from_bands(ax, qpath, bands, [0, 1], ["A", "B"], weights, 0, 400, Ny=401, Nx_uniform=50, sigma=5, dy=0)
    assert mmap.shape == (401, 50) and grid_x.shape == (50,) and grid_y.shape == (401,)
    np.testing.assert_allclose(grid_y[mmap.argmax(axis=0)], 100)  # only the weighted band shows
    np.testing.assert_allclose(mmap.max(), 1)  # unnormalized Gaussian: peak = weight
    assert mmap[np.abs(grid_y - 300) < 1].max() < 1e-12


def test_resample_handles_segment_joints():
    x = np.array([0.0, 1.0, 1.0, 2.0])  # repeated point at the joint of two segments
    y = np.array([[0.0], [1.0], [1.0], [3.0]])
    np.testing.assert_allclose(_resample(x, y, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))[:, 0], [0, 0.5, 1, 2, 3])
