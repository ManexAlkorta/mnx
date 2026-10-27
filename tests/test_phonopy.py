"""
Tests of DynMatrix.from_phonopy on the VASP + phonopy example in Tutorial/dyns/phonopy.

The reference is phonopy itself, which interpolates the same force constants.
"""

import numpy as np
import phonopy
import pytest

import mnx
from conftest import DYNS

FOLDER = DYNS / "phonopy"
THZ_TO_CM = 33.35641

# Newer phonopy deprecates get_dynamical_matrix_at_q/get_frequencies; both still work.
pytestmark = pytest.mark.filterwarnings("ignore:.*is deprecated:DeprecationWarning")


@pytest.fixture(scope="module")
def phonopy_dyn():
    return mnx.DynMatrix.from_phonopy(str(FOLDER), qgrid=[2, 2, 2])


def test_loads_from_any_working_directory(tmp_path, monkeypatch):
    """FORCE_SETS is read from the folder, not from the current directory."""
    monkeypatch.chdir(tmp_path)
    dyn = mnx.DynMatrix.from_phonopy(str(FOLDER), qgrid=[2, 2, 2])
    assert dyn.Nqirr == 8 and dyn.Nqpoints == 8


def test_acoustic_modes_vanish_at_gamma(phonopy_dyn):
    np.testing.assert_allclose(np.abs(phonopy_dyn.DynQs[0].frequencies[0, :3]), 0, atol=1e-2)


@pytest.mark.parametrize("q", [[0, 0, 0], [0.5, 0, 0], [0.25, 0, 0], [0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
def test_interpolation_matches_phonopy(phonopy_dyn, q):
    """On and off the grid, the atomic-gauge interpolation reproduces phonopy (< 1e-2 cm^-1)."""
    reference = phonopy.load(str(FOLDER / "phonopy.yaml"), force_sets_filename=str(FOLDER / "FORCE_SETS"))
    bands, _, _ = phonopy_dyn.get_bands([[q, q]], N=1, gauge="atomic")
    np.testing.assert_allclose(bands[0], reference.get_frequencies(q) * THZ_TO_CM, atol=1e-2)
