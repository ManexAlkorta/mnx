"""
Tests of the Fourier interpolation of the force constants (DynMatrix.get_bands / get_dynq).

The reference is QuantumESPRESSO: ``tests/data/RbH12_3x3x3`` holds the q2r.x and matdyn.x
inputs and the ``freq`` file that matdyn.x produced from the same dyn files that mnx reads
(``Tutorial/dyns/RbH12/3x3x3/hessian*``).
"""

import numpy as np
import pytest

import mnx.utils.cell as _cell
from conftest import DATA, read_matdyn_freq, read_matdyn_path

QE_DIR = DATA / "RbH12_3x3x3"

# matdyn.x prints frequencies with 4 decimals; mnx agrees to ~2e-3 cm^-1.
QE_ATOL = 1e-2  # cm^-1


def _signed(frequencies):
    """Complex frequencies to real ones, imaginary shown as negative."""
    frequencies = np.asarray(frequencies)
    return np.where(np.square(frequencies).real < 0, -np.abs(frequencies), frequencies.real)


@pytest.fixture(scope="module")
def qe_reference():
    q_cart, frequencies = read_matdyn_freq(QE_DIR / "freq")
    points, nsteps = read_matdyn_path(QE_DIR / "matdyn.in")
    return q_cart, frequencies, points, nsteps


@pytest.fixture(scope="module")
def mnx_bands(rbh12, qe_reference):
    """mnx bands along the matdyn.x path, with the same number of points per segment."""
    _, _, points, nsteps = qe_reference
    N = nsteps[0]
    assert all(n == N for n in nsteps[:-1]), "the test assumes equally sampled segments"
    k_inpath = np.stack([points[:-1], points[1:]], axis=1)
    return rbh12.get_bands(k_inpath=k_inpath, N=N, gauge="atomic"), N


def _to_qe_points(array, nsegments, N):
    """
    Drop the points duplicated at the segment joints of a mnx path.

    mnx samples every segment with N + 1 points, matdyn.x with N points plus the final one.
    """
    array = np.asarray(array)
    keep = [s * (N + 1) + j for s in range(nsegments) for j in range(N)] + [nsegments * (N + 1) - 1]
    return array[keep]


def test_path_matches_matdyn_qpoints(rbh12, qe_reference):
    """The q-points mnx samples are the ones matdyn.x used (sanity check of the comparison)."""
    q_cart, _, points, nsteps = qe_reference
    N = nsteps[0]
    k_inpath = np.stack([points[:-1], points[1:]], axis=1)
    qpath, _, _ = rbh12._build_qpath(k_inpath, N)
    qpath = _to_qe_points(qpath, len(k_inpath), N)
    qe_cryst = _cell.cart2cryst(q_cart / rbh12._alat, rbh12.structure.rcell)
    np.testing.assert_allclose(qpath, qe_cryst, atol=1e-5)


def test_bands_match_quantum_espresso(mnx_bands, qe_reference):
    """Interpolated frequencies along a 900-point path agree with matdyn.x."""
    _, qe_frequencies, points, _ = qe_reference
    (bands, _, _), N = mnx_bands
    bands = _to_qe_points(bands, len(points) - 1, N)
    assert bands.shape == qe_frequencies.shape
    np.testing.assert_allclose(bands, qe_frequencies, atol=QE_ATOL, rtol=0)


@pytest.mark.parametrize("gauge", ["atomic", "lattice"])
def test_grid_points_reproduce_dyn_files(rbh12, gauge):
    """At the q-points of the grid the interpolation must give back the input spectra."""
    for dynq in rbh12.DynQs:
        for qi, q in enumerate(dynq.qpoints):
            interpolated = rbh12.get_dynq(q, gauge=gauge)
            np.testing.assert_allclose(
                _signed(interpolated.frequencies[0]), _signed(dynq.frequencies[qi]), atol=1e-3, rtol=0,
                err_msg=f"q = {q}, gauge = {gauge}",
            )


@pytest.mark.parametrize("gauge", ["atomic", "lattice"])
def test_get_dynq_consistent_with_get_bands(crsb, gauge):
    """get_dynq and get_bands interpolate to the same spectrum."""
    k_inpath = [[[0, 0, 0], [0.5, 0, 0]], [[0.5, 0, 0], [1 / 3, 1 / 3, 0.5]]]
    bands, _, _ = crsb.get_bands(k_inpath=k_inpath, N=4, gauge=gauge)
    qpath, _, _ = crsb._build_qpath(k_inpath, 4)
    for q, band in zip(qpath, bands):
        np.testing.assert_allclose(_signed(crsb.get_dynq(q, gauge=gauge).frequencies[0]), band, atol=1e-8)


@pytest.mark.parametrize("N", [1, 7, 20, 30, 100])
def test_band_path_sampling(crsb, N):
    """Every N gives N + 1 points per segment with ticks at the segment ends (N=30 used to crash)."""
    k_inpath = [[[0, 0, 0], [0.5, 0, 0]], [[0.5, 0, 0], [0.5, 0, 0.5]]]
    bands, cart_qpath, xticks = crsb.get_bands(k_inpath=k_inpath, N=N)
    assert bands.shape == (2 * (N + 1), crsb.structure.Natoms * 3)
    assert cart_qpath[0] == 0 and np.all(np.diff(cart_qpath) >= 0)
    np.testing.assert_allclose(xticks, [0, cart_qpath[N], cart_qpath[-1]])
