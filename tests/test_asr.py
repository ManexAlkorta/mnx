"""
Tests of the acoustic sum rule (DynMatrix.apply_asr).

References: QuantumESPRESSO 6.7 matdyn.x with asr='no' and asr='simple' on CrSb
(tests/data/CrSb_2x2x2, no effective charges) and MnTe (tests/data/MnTe_2x2x2, with LO-TO
splitting), along the path of test_loto.py.
"""

import numpy as np
import pytest

import mnx
from conftest import DATA, read_matdyn_freq
from test_loto import N, PATH


@pytest.fixture(params=["crsb", "mnte"])
def dyn_with_asr(request):
    dyn = request.getfixturevalue(request.param).copy()
    dyn.apply_asr("simple")
    return request.param, dyn


def _acoustic_at_gamma(dyn, direction=None):
    return np.sort(np.abs(dyn.get_dynq([0, 0, 0], direction=direction).frequencies[0]))[:3]


@pytest.mark.parametrize(
    "name, data, asr",
    [("crsb", "CrSb_2x2x2", "no"), ("crsb", "CrSb_2x2x2", "simple"), ("mnte", "MnTe_2x2x2", "simple")],
)
def test_bands_match_quantum_espresso(name, data, asr, request):
    """matdyn.x prints 4 decimals; the agreement is ~1e-4 cm^-1."""
    dyn = request.getfixturevalue(name).copy()
    if asr == "simple":
        dyn.apply_asr("simple")
    _, reference = read_matdyn_freq(DATA / data / f"freq_asr_{asr}")
    bands, _, _ = dyn.get_bands(PATH, N)
    np.testing.assert_allclose(bands, reference, atol=1e-3)


def test_acoustic_modes_vanish_at_gamma(dyn_with_asr):
    name, dyn = dyn_with_asr
    np.testing.assert_allclose(_acoustic_at_gamma(dyn), 0, atol=1e-3)
    if name == "mnte":  # with the LO-TO splitting too
        for direction in [[1, 0, 0], [0, 0, 1], [1, 1, 1]]:
            np.testing.assert_allclose(_acoustic_at_gamma(dyn, direction), 0, atol=1e-3)


def test_row_sums_vanish(dyn_with_asr):
    _, dyn = dyn_with_asr
    np.testing.assert_allclose(dyn.phiR.sum(axis=1), 0, atol=1e-12)


def test_kept_by_write_and_apply_nac(mnte, tmp_path):
    dyn = mnte.copy()
    dyn.apply_asr("simple")
    bands, _, _ = dyn.get_bands(PATH, 4)

    dyn.write(str(tmp_path / "dyn"))
    reread = mnx.DynMatrix.from_file(str(tmp_path / "dyn"), qgrid=[2, 2, 2], Nqirr=4)
    np.testing.assert_allclose(_acoustic_at_gamma(reread), 0, atol=1e-3)
    np.testing.assert_allclose(reread.get_bands(PATH, 4)[0], bands, atol=1e-4)

    dyn.apply_nac = False
    np.testing.assert_allclose(_acoustic_at_gamma(dyn), 0, atol=1e-3)
    dyn.apply_nac = True
    # ~1e-5 cm^-1 acoustic modes: omega = sqrt(lambda) magnifies round-off of lambda ~ 0.
    np.testing.assert_allclose(dyn.get_bands(PATH, 4)[0], bands, atol=1e-4)


def test_grid_modes_updated(mnte):
    """The modes of the Gamma star (used by distort_structure, get_Pmn...) satisfy the rule too."""
    dyn = mnte.copy()
    assert np.abs(dyn.DynQs[0].frequencies[0][:3]).max() > 1  # 2.9 and 7.2 cm^-1 before
    dyn.apply_asr("simple")
    np.testing.assert_allclose(np.abs(dyn.DynQs[0].frequencies[0][:3]), 0, atol=1e-3)


def test_unknown_rule(crsb):
    with pytest.raises(ValueError):
        crsb.copy().apply_asr("crystal")
