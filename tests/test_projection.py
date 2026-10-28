"""Tests of the phonon mode basis and of DynMatrix.project_structure (inverse of distort_structure)."""

import numpy as np
import pytest

import mnx
from conftest import DYNS


@pytest.fixture(scope="module")
def phonopy_dyn():
    return mnx.DynMatrix.from_phonopy(str(DYNS / "phonopy"), qgrid=[2, 2, 2])


def _shuffled(structure, seed=0):
    order = np.random.default_rng(seed).permutation(structure.Natoms)
    return mnx.Structure.from_data(structure.cell, structure.atom_coords[order], structure.atom_species[order])


@pytest.mark.filterwarnings("ignore:.*is deprecated:DeprecationWarning")
@pytest.mark.parametrize("name", ["crsb", "rbh12", "phonopy_dyn"])
def test_modes_are_orthonormal(name, request):
    """Polarization vectors are orthonormal at every q-point, also inside degenerate subspaces."""
    dyn = request.getfixturevalue(name)
    for dynq in dyn.DynQs:
        for polvecs in dynq.polvecs:
            np.testing.assert_allclose(polvecs.conj() @ polvecs.T, np.eye(len(polvecs)), atol=1e-12)


def test_modes_are_orthonormal_after_symmetrize():
    dyn = mnx.DynMatrix.from_file(str(DYNS / "CsV3Sb5" / "hessianv4_"), qgrid=[1, 1, 1], Nqirr=1)
    dyn.Symmetrize(symprec=1e-3)
    polvecs = dyn.DynQs[0].polvecs[0]
    np.testing.assert_allclose(polvecs.conj() @ polvecs.T, np.eye(len(polvecs)), atol=1e-12)


def test_self_conjugate_modes_are_real(crsb):
    """At Gamma, A, M and L (2q = G) the modes are chosen real."""
    for dynq in crsb.DynQs:
        assert np.all(dynq.polvecs.imag == 0)


def test_time_reversal_partners_are_conjugate(rbh12):
    """When -q is in the same star its modes are the complex conjugates of those at q."""
    dynq = rbh12.DynQs[2]
    np.testing.assert_allclose(dynq.qpoints[0] + dynq.qpoints[1], 0, atol=1e-6)
    np.testing.assert_array_equal(dynq.polvecs[1], dynq.polvecs[0].conj())


def test_characters_are_integers(crsb):
    """With orthonormal modes the little-group characters of CrSb are integers."""
    crsb.get_Pmn()
    for dynq in crsb.DynQs:
        dynq.compute_traces()
        np.testing.assert_allclose(dynq.traces, np.round(dynq.traces.real), atol=1e-10)


@pytest.mark.parametrize(
    "Q, modes",
    [
        ([[10, 0, 0], [0, 10, 10]], [[2, 0], [3, 0]]),  # M and L stars of CrSb
        ([[3, -2, 7], [1, 0, 0], [5]], [[2, 4], [3, 9], [0, 6]]),
        ([[0.5], [-4]], [[1, 11], [0, 3]]),
    ],
)
def test_project_inverts_distort(crsb, Q, modes):
    distorted = crsb.distort_structure(Q=Q, modes=modes)
    projected = crsb.project_structure(_shuffled(distorted), modes=modes)
    assert len(projected) == len(Q)
    for amplitudes, expected in zip(projected, Q):
        np.testing.assert_allclose(amplitudes, expected, atol=1e-9)


def test_project_in_smaller_supercell(crsb):
    """The supercell is read from the cell: here 2x2x1, where only the M star fits."""
    distorted = crsb.distort_structure(Q=[[1, 2, 0]], modes=[[2, 5]], mod=[2, 2, 1])
    np.testing.assert_allclose(crsb.project_structure(distorted, modes=[[2, 5]])[0], [1, 2, 0], atol=1e-9)


def test_unselected_modes_do_not_leak(crsb):
    """Modes are orthogonal: a distortion along one mode projects to zero on the others."""
    distorted = crsb.distort_structure(Q=[[10, 0, 0]], modes=[[2, 0]])
    others = [[2, mode] for mode in range(1, 12)] + [[0, 5], [3, 0]]
    for amplitudes in crsb.project_structure(distorted, modes=others):
        np.testing.assert_allclose(amplitudes, 0, atol=1e-9)


@pytest.mark.parametrize("Q", [[2, 2], [2 + 1j, 2 - 1j], [-3j, 3j]])
def test_time_reversal_pairs_round_trip(rbh12, Q):
    """With q and -q in one star, any real distortion (Q_-q = conj(Q_q)) comes back unchanged."""
    distorted = rbh12.distort_structure(Q=[Q], modes=[[2, 10]])
    np.testing.assert_allclose(rbh12.project_structure(_shuffled(distorted), modes=[[2, 10]])[0], Q, atol=1e-9)


@pytest.mark.parametrize(
    "name, Q, modes",
    [
        ("rbh12", [[4, 0]], [[2, 10]]),  # q without its -q partner
        ("rbh12", [[2, 2 + 1j]], [[2, 10]]),  # Q_-q != conj(Q_q)
        ("crsb", [[10j, 0, 0]], [[2, 0]]),  # complex amplitude on a real (zone-boundary) mode
    ],
)
def test_imaginary_distortion_raises(name, Q, modes, request):
    with pytest.raises(ValueError, match="imaginary displacement"):
        request.getfixturevalue(name).distort_structure(Q=Q, modes=modes)


def test_projection_is_independent_per_mode(rbh12):
    """Each amplitude depends only on its own mode, not on which other modes are selected."""
    reference = rbh12.structure.expand_structure([3, 3, 3])
    noise = 0.01 * np.random.default_rng(1).standard_normal(reference.atom_coords.shape)
    noisy = mnx.Structure.from_data(reference.cell, reference.atom_coords + noise, reference.atom_species)
    alone = rbh12.project_structure(noisy, modes=[[2, 10]])[0]
    together = rbh12.project_structure(noisy, modes=[[3, 1], [2, 10], [0, 5]])[1]
    np.testing.assert_allclose(alone, together, atol=1e-14)
    np.testing.assert_allclose(alone[1], alone[0].conj(), atol=1e-12)  # real displacement


def test_atom_beyond_cutoff_raises(crsb):
    distorted = crsb.distort_structure(Q=[[10, 0, 0]], modes=[[2, 0]])
    distorted.atom_coords = distorted.atom_coords.copy()
    distorted.atom_coords[5] += [0.3, 0, 0]
    with pytest.raises(ValueError, match="No atom within 0.2 A of site"):
        crsb.project_structure(distorted, modes=[[2, 0]])
    crsb.project_structure(distorted, modes=[[2, 0]], cutoff=0.5)  # a larger cutoff accepts it


def test_wrong_cell_raises(crsb):
    strained = crsb.distort_structure(Q=[[10, 0, 0]], modes=[[2, 0]])
    strained.cell = strained.cell * 1.01
    with pytest.raises(ValueError, match="not a diagonal supercell"):
        crsb.project_structure(strained, modes=[[2, 0]])


def test_incommensurate_star_raises(crsb):
    distorted = crsb.distort_structure(Q=[[1, 0, 0]], modes=[[2, 0]], mod=[2, 2, 1])
    with pytest.raises(ValueError, match="not commensurate"):
        crsb.project_structure(distorted, modes=[[3, 0]])  # L star needs 2 cells along c
    with pytest.raises(ValueError, match="not commensurate"):
        crsb.distort_structure(Q=[[1, 0, 0]], modes=[[3, 0]], mod=[2, 2, 1])
