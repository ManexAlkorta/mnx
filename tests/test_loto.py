"""
Tests of the non-analytic correction (long-range dipole-dipole term and LO-TO splitting).

The data is MnTe on a 2x2x2 grid (``Tutorial/dyns/MnTe``), whose Gamma file has the
dielectric tensor and the Born effective charges. The references in ``tests/data/MnTe_2x2x2``
are QuantumESPRESSO's matdyn.x and CellConstructor's ForceTensor.Tensor2, along a path that
reaches Gamma from four directions.
"""

import numpy as np
import pytest

import mnx
import mnx.utils.consts as _consts
from conftest import DATA, DYNS, read_matdyn_freq

MNTE_DIR = DATA / "MnTe_2x2x2"
MNTE_DYN = str(DYNS / "MnTe" / "dyn")

G, M, K, A, L, H = [0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0.5], [0.5, 0, 0.5], [1 / 3, 1 / 3, 0.5]
PATH = np.array([(G, M), (M, K), (K, G), (G, A), (A, L), (L, H), (H, A), (L, G)], dtype=float)
N = 20  # intervals per segment in the references

# Analytic (no LO-TO) frequencies at Gamma written by ph.x in Tutorial/dyns/MnTe/dyn1, cm^-1.
GAMMA_ANALYTIC = [2.863655, 2.863655, 7.249403, 93.208951, 93.208951, 96.377904, 96.377904,
                  121.133581, 128.554573, 134.932051, 134.932051, 180.894170]


def _numpy_nonanalytic(qhat_cart, epsilon, zeff, volume_bohr):
    """Independent implementation of Eq. (60) of Gonze & Lee, PRB 55, 10355 (1997), in Ry/bohr^2."""
    qz = np.einsum("i,aij->aj", qhat_cart, zeff)
    return 4 * np.pi * 2 * np.einsum("ai,bj->abij", qz, qz) / (qhat_cart @ epsilon @ qhat_cart) / volume_bohr


def _optical(frequencies):
    """Drop the three acoustic modes, which are not exactly zero because no acoustic sum rule is imposed."""
    return np.sort(np.real(frequencies))[3:]


def _assert_matches_ph(frequencies, expected):
    """
    Compare with frequencies printed by ph.x: mnx's Ry -> cm^-1 factor differs from QE's by 4e-7
    (relative), and the ~3 cm^-1 acoustic modes amplify tiny errors of omega^2, so they get an
    absolute tolerance.
    """
    frequencies, expected = np.real(frequencies), np.asarray(expected)
    np.testing.assert_allclose(frequencies[:3], expected[:3], atol=1e-3)
    np.testing.assert_allclose(frequencies[3:], expected[3:], rtol=1e-6)


def test_reads_epsilon_and_zeff(mnte):
    gamma = mnte.DynQs[0]
    assert gamma.epsilon[2, 2] == pytest.approx(14.443758680191)
    assert gamma.zeff[0, 0, 0] == pytest.approx(2.815620158481)  # "E-U" block, not "U-E" (2.8179)
    np.testing.assert_allclose(gamma.zeff[2], gamma.zeff[3], atol=1e-10)  # the two Te are equivalent
    assert all(dynq.zeff is None and dynq.epsilon is None for dynq in mnte.DynQs[1:])
    assert mnte.apply_nac
    np.testing.assert_allclose(mnte.epsilon, gamma.epsilon)


def test_charge_sum_rule(mnte):
    raw = mnte.DynQs[0].zeff
    assert np.abs(raw.sum(axis=0)).max() > 1e-4  # the DFPT charges are not exactly neutral
    np.testing.assert_allclose(mnte.zeff.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(mnte.zeff, raw - raw.mean(axis=0), atol=1e-12)

    no_rule = mnx.DynMatrix.from_file(MNTE_DYN, qgrid=[2, 2, 2], Nqirr=4, zasr="no")
    np.testing.assert_allclose(no_rule.zeff, raw)
    with pytest.raises(ValueError):
        mnx.DynMatrix.from_file(MNTE_DYN, qgrid=[2, 2, 2], Nqirr=4, zasr="crystal")


def test_hexagonal_cell_from_ibrav(mnte):
    """dyn files with ibrav=4 (no "Basis vectors" block) give QE's hexagonal cell."""
    a = 7.9168599 * _consts.bohr2angstroms
    expected = a * np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 1.5881760]])
    np.testing.assert_allclose(mnte.structure.cell, expected, atol=1e-6)
    np.testing.assert_allclose(mnte.structure.atom_coords[1], [0, 0, 0.7940879777 * a], atol=1e-6)


def test_write_keeps_charges(mnte, tmp_path):
    mnte.write(str(tmp_path / "dyn"))
    reread = mnx.DynMatrix.from_file(str(tmp_path / "dyn"), qgrid=[2, 2, 2], Nqirr=4)
    np.testing.assert_allclose(reread.DynQs[0].zeff, mnte.DynQs[0].zeff, atol=1e-11)
    np.testing.assert_allclose(reread.epsilon, mnte.epsilon, atol=1e-11)
    q = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(reread.get_dynq(q).frequencies, mnte.get_dynq(q).frequencies, atol=1e-6)


@pytest.mark.parametrize("gauge", ["lattice", "atomic"])
def test_grid_points_reproduced(mnte, gauge):
    """Short-range Fourier interpolation plus the long-range term gives back the dyn files on the grid."""
    for dynq in mnte.DynQs:
        for qi, q in enumerate(dynq.qpoints):
            interpolated = mnte.get_dynq(q, gauge=gauge)
            np.testing.assert_allclose(_optical(interpolated.frequencies[0]), _optical(dynq.frequencies[qi]), atol=1e-6)
            if gauge == "lattice":
                np.testing.assert_allclose(interpolated.phiqs[0], dynq.phiqs[qi], atol=1e-8)  # dyn files have 8 decimals


def test_gamma_without_direction_is_analytic(mnte):
    _assert_matches_ph(mnte.get_dynq([0, 0, 0]).frequencies[0], GAMMA_ANALYTIC)
    assert mnte.get_dynq([0, 0, 0]).direction is None


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 0, 1], [1, 1, 1], [0.3, -0.2, 0.7]])
def test_nonanalytic_term_matches_numpy(mnte, direction):
    qhat_cart = np.asarray(direction, dtype=float) @ mnte.structure.rcell
    volume = abs(np.linalg.det(mnte.structure.cell)) * _consts.angstroms2bohr**3
    expected = _numpy_nonanalytic(qhat_cart, mnte.epsilon, mnte.zeff, volume)
    np.testing.assert_allclose(mnte._nonanalytic(direction), expected, rtol=1e-12, atol=1e-14)

    analytic = mnte.get_dynq([0, 0, 0])
    with_nac = mnte.get_dynq([0, 0, 0], direction=direction)
    np.testing.assert_allclose(with_nac.phiqs[0] - analytic.phiqs[0], expected, atol=1e-12)
    np.testing.assert_allclose(with_nac.direction, direction)


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 0, 1], [1, 1, 1]])
def test_loto_limit_is_continuous(mnte, direction):
    """omega(q -> 0 along u) tends to the Gamma frequencies with the non-analytic term along u."""
    direction = np.asarray(direction, dtype=float)
    gamma = mnte.get_dynq([0, 0, 0], direction=direction).frequencies[0]
    near = mnte.get_dynq(1e-5 * direction).frequencies[0]
    np.testing.assert_allclose(_optical(near), _optical(gamma), atol=1e-5)


def test_loto_direction_dependence(mnte):
    """In-plane q splits the E1u mode (134.93 -> 179.83), q along z the A2u mode (128.55 -> 168.14)."""
    in_plane = mnte.get_dynq([0, 0, 0], direction=[1, 0, 0]).frequencies[0].real
    along_z = mnte.get_dynq([0, 0, 0], direction=[0, 0, 1]).frequencies[0].real
    np.testing.assert_allclose(in_plane[[8, 9, 10]], [128.5546, 134.9321, 179.8320], atol=1e-3)
    np.testing.assert_allclose(along_z[[8, 9, 10]], [134.9321, 134.9321, 168.1360], atol=1e-3)
    # Optical modes without a dipole are not affected. (The acoustic modes are, slightly, as in
    # matdyn.x: without the acoustic sum rule they are not pure translations.)
    np.testing.assert_allclose(in_plane[3:8], GAMMA_ANALYTIC[3:8], rtol=1e-6)
    np.testing.assert_allclose(along_z[3:8], GAMMA_ANALYTIC[3:8], rtol=1e-6)


@pytest.mark.parametrize("q", [[0.13, 0.07, 0.21], [0.5, 0, 0], [-0.2, 0.4, 0.5]])
def test_long_range_gauges(mnte, q):
    """The atomic-gauge long-range term is the lattice-gauge one times exp(2 pi i q.(tau_b - tau_a))."""
    q = np.asarray(q, dtype=float)
    phases = np.exp(2j * np.pi * (mnte.structure.atom_coords @ (q @ mnte.structure.rcell)))
    lattice = mnte._long_range(q, "lattice")
    atomic = mnte._long_range(q, "atomic")
    np.testing.assert_allclose(atomic, lattice * np.outer(phases.conj(), phases)[:, :, None, None], atol=1e-15)
    # Hermitian, as a block of a dynamical matrix must be.
    matrix = lattice.transpose(0, 2, 1, 3).reshape(12, 12)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


def test_apply_nac_switch(mnte, rbh12):
    dyn = mnte.copy()
    with_nac, _, _ = dyn.get_bands(PATH, 4)
    dyn.apply_nac = False
    assert dyn.zeff is not None  # the charges are kept
    _assert_matches_ph(dyn.get_dynq([0, 0, 0], direction=[0, 0, 1]).frequencies[0], GAMMA_ANALYTIC)
    without_nac, _, _ = dyn.get_bands(PATH, 4)
    assert np.abs(with_nac - without_nac).max() > 10  # the LO modes move by tens of cm^-1
    dyn.apply_nac = True
    np.testing.assert_allclose(dyn.get_bands(PATH, 4)[0], with_nac, atol=1e-10)

    assert not rbh12.apply_nac and rbh12.zeff is None
    with pytest.raises(ValueError):
        rbh12.copy().apply_nac = True


def test_bands_match_quantum_espresso(mnte):
    """matdyn.x with zasr='simple' and asr='no' (tests/data/MnTe_2x2x2)."""
    q_cart, reference = read_matdyn_freq(MNTE_DIR / "freq")
    bands, _, _ = mnte.get_bands(PATH, N)
    qpath, _, _ = mnte._build_qpath(PATH, N)
    alat = np.linalg.norm(mnte.structure.cell[0])
    np.testing.assert_allclose(qpath @ mnte.structure.rcell * alat, q_cart, atol=1e-5)  # same q-points
    # matdyn.x prints 4 decimals; the agreement is ~1e-4 cm^-1.
    np.testing.assert_allclose(bands, reference, atol=1e-3)


def test_bands_match_cellconstructor():
    """CellConstructor's ForceTensor.Tensor2, charges without sum rule (tests/data/MnTe_2x2x2)."""
    reference = np.loadtxt(MNTE_DIR / "cc_freq.txt")
    dyn = mnx.DynMatrix.from_file(MNTE_DYN, qgrid=[2, 2, 2], Nqirr=4, zasr="no")
    bands, _, _ = dyn.get_bands(PATH, N)
    qpath, _, _ = dyn._build_qpath(PATH, N)
    np.testing.assert_allclose(qpath, reference[:, :3], atol=1e-7)
    np.testing.assert_allclose(bands, reference[:, 3:], atol=1e-4)


def test_symmetrize_keeps_loto(mnte):
    """Symmetrize acts on the short-range force constants and adds the long-range term back on the grid."""
    dyn = mnte.copy()
    dyn.Symmetrize(symprec=1e-3)
    for direction in [None, [1, 0, 0], [0, 0, 1]]:
        np.testing.assert_allclose(
            _optical(dyn.get_dynq([0, 0, 0], direction=direction).frequencies[0]),
            _optical(mnte.get_dynq([0, 0, 0], direction=direction).frequencies[0]),
            atol=1e-3,
        )
    for dynq, original in zip(dyn.DynQs, mnte.DynQs):
        np.testing.assert_allclose(dynq.phiqs, original.phiqs, atol=1e-6)


@pytest.mark.parametrize("gauge", ["atomic", "lattice"])
@pytest.mark.parametrize("G", [[1, 0, 0], [0, 1, -1], [2, -1, 3]])
def test_loto_at_equivalent_gamma(mnte, G, gauge):
    """A Gamma point outside the first zone (q = G) has the same LO-TO splitting as q = 0."""
    for direction in [[1, 0, 0], [0, 0, 1], [1, 1, 1]]:
        np.testing.assert_allclose(
            mnte.get_dynq(G, gauge=gauge, direction=direction).frequencies[0].real,
            mnte.get_dynq([0, 0, 0], gauge=gauge, direction=direction).frequencies[0].real,
            atol=1e-8,
        )
    # Gamma -> M -> Gamma' is mirror symmetric, also at its end points.
    bands, _, _ = mnte.get_bands([[[0, 0, 0], [0.5, 0, 0]], [[0.5, 0, 0], [1, 0, 0]]], 10, gauge=gauge)
    np.testing.assert_allclose(bands[:11], bands[11:][::-1], atol=1e-8)
