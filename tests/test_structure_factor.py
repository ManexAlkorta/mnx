"""
Tests of the one-phonon structure factor (DynMatrix.get_bands_with_SF).

The reference is Eq. (2) of the Supplementary Information of arXiv:2603.25317,
F = sum_a (K . eps_a(k)) exp(i K . R_a) / sqrt(2 M_a w), with the periodic (QuantumESPRESSO)
polarization vectors and K = Q + k. Three independent checks:

* Eq. (2) evaluated literally on the polarization vectors of the dyn files (grid points);
* S depends only on K, not on how it is split into Q + k (k beyond the first zone);
* a real-space supercell built from the same force constants, off the grid, where only the
  real atomic displacements enter and no phase convention is involved.
"""

import itertools

import numpy as np
import pytest

import mnx
import mnx.utils.cell as _cell
from mnx.dyn_matrix import _eigenvalues_to_cm
from conftest import DYNS

G_VECTORS = [(0, 0, 3), (3, 0, 0), (1, 0, 3), (2, -1, 1)]


def _sf_at(dyn, k, Q):
    """Frequencies and structure factor of get_bands_with_SF at a single k."""
    k = np.asarray(k, dtype=float)
    bands, _, _, sf = dyn.get_bands_with_SF([[k, k]], 1, Q)
    return bands[0], sf[0]


def _by_frequency(frequencies, values, centers, tol=1e-2):
    """Sum over (nearly) degenerate modes, the quantity that does not depend on the basis."""
    return np.array([values[np.abs(frequencies - c) < tol].sum() for c in centers])


def _assert_same(frequencies, values, ref_frequencies, ref_values, rtol):
    centers = np.unique(np.round(ref_frequencies[ref_frequencies > 1], 2))
    expected = _by_frequency(ref_frequencies, ref_values, centers)
    got = _by_frequency(frequencies, values, centers)
    np.testing.assert_allclose(got, expected, atol=rtol * expected.max())


def _eq2(frequencies, polvecs, masses, positions, K_cart):
    """Eq. (2) of the supplement, |F|^2, for polarization vectors given as rows [mode, 3*atom]."""
    eps = polvecs.reshape(len(frequencies), -1, 3)
    weights = np.exp(2j * np.pi * (positions @ K_cart)) / np.sqrt(2 * masses)
    F = np.einsum("man,n->m", weights[None, :, None] * eps, K_cart)
    return np.abs(F) ** 2 / np.where(frequencies > 1, frequencies, np.inf)


def test_matches_eq2_on_dyn_file_vectors(crsb):
    """Literal Eq. (2) with the QE polarization vectors of the dyn files, at every grid point."""
    for dynq in crsb.DynQs:
        for qi, q in enumerate(dynq.qpoints):
            if np.allclose(q, 0):
                continue  # acoustic modes at zero frequency
            for G in G_VECTORS:
                K_cart = _cell.cryst2cart(q + np.asarray(G, float), crsb.structure.rcell)
                w_ref = dynq.frequencies[qi].real
                ref = _eq2(w_ref, dynq.polvecs[qi], crsb.structure.masses, crsb.structure.atom_coords, K_cart)
                w, sf = _sf_at(crsb, q, G)
                _assert_same(w, sf, w_ref, ref, rtol=1e-6)


@pytest.mark.parametrize("name", ["crsb", "mnte"])
def test_depends_only_on_total_momentum(name, request):
    """(k, Q) and (k - G, Q + G) give the same S, so paths may cross into the next zone."""
    dyn = request.getfixturevalue(name)
    for k, Q in [((0.1, 0, 0), (0, 0, 3)), ((0.3, 0.1, 0), (3, 0, 0)), ((0.37, -0.11, 0.23), (2, -1, 1))]:
        w0, sf0 = _sf_at(dyn, k, Q)
        for G in [(1, 0, 0), (0, 1, -1), (-1, 0, 1)]:
            w1, sf1 = _sf_at(dyn, np.add(k, G), np.subtract(Q, G))
            _assert_same(w1, sf1, w0, sf0, rtol=1e-9)


def _model_terms(dyn):
    """Real-space force constants Phi(a0; b T) behind the interpolation: phiR spread over its
    shortest (Wigner-Seitz) images, as the Fortran routine ft_phir does."""
    prim = dyn.structure
    inv = np.linalg.inv(prim.cell)
    frac_sc = dyn.super_structure.atom_coords @ inv
    tau = prim.atom_coords @ inv
    shifts = np.array([np.multiply(s, dyn.qgrid) for s in itertools.product([-1, 0, 1], repeat=3)])
    terms = []
    for a in range(prim.Natoms):
        for B in range(dyn.super_structure.Natoms):
            b = B % prim.Natoms
            images = frac_sc[B] - frac_sc[a] + shifts
            dist = np.linalg.norm(images @ prim.cell, axis=1)
            shortest = images[np.abs(dist - dist.min()) < 1e-5]
            for image in shortest:
                T = np.round(image - (tau[b] - tau[a])).astype(int)
                terms.append((a, b, T, dyn.phiR[a, B] / len(shortest)))
    return terms


def _supercell_sf(dyn, terms, N, K):
    """Eq. (2) for the real normal modes of an N supercell of the force-constant model."""
    prim = dyn.structure
    nat = prim.Natoms
    cells = [np.array(c) for c in itertools.product(*[range(n) for n in N])]
    index = {tuple(c): i for i, c in enumerate(cells)}
    nbig = nat * len(cells)
    phi = np.zeros([nbig, 3, nbig, 3])
    for ci, C in enumerate(cells):
        for a, b, T, block in terms:
            phi[ci * nat + a, :, index[tuple(np.mod(C + T, N))] * nat + b, :] += block
    masses = np.tile(prim.masses, len(cells))
    m3 = np.repeat(masses, 3)
    eigenvalues, vectors = np.linalg.eigh(phi.reshape(3 * nbig, 3 * nbig) / np.sqrt(np.outer(m3, m3)))
    frequencies = _eigenvalues_to_cm(eigenvalues).real
    positions = np.concatenate([prim.atom_coords + C @ prim.cell for C in cells])
    K_cart = _cell.cryst2cart(K, prim.rcell)
    return frequencies, _eq2(frequencies, vectors.T, masses, positions, K_cart) / len(cells)


@pytest.mark.parametrize("N, k", [((4, 1, 1), (0.25, 0, 0)), ((1, 1, 3), (0, 0, 1 / 3))])
def test_matches_real_space_supercell(crsb, N, k):
    """Off the grid (k and -k inequivalent), against the real modes of a larger supercell."""
    terms = _model_terms(crsb)
    for Q in G_VECTORS:
        w_ref, ref = _supercell_sf(crsb, terms, np.array(N), np.add(Q, k))
        w, sf = _sf_at(crsb, k, Q)
        _assert_same(w_ref, ref, w, sf, rtol=1e-9)
