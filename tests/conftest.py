import re
from pathlib import Path

import numpy as np
import pytest

import mnx

ROOT = Path(__file__).resolve().parents[1]
DYNS = ROOT / "Tutorial" / "dyns"
DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def rbh12():
    """RbH12 phonons on a 3x3x3 grid (6 irreducible q-points), the input of the QE reference."""
    return mnx.DynMatrix.from_file(str(DYNS / "RbH12" / "3x3x3" / "hessian"), qgrid=[3, 3, 3], Nqirr=6)


@pytest.fixture(scope="session")
def crsb():
    """CrSb phonons on a 2x2x2 grid (4 irreducible q-points)."""
    return mnx.DynMatrix.from_file(str(DYNS / "CrSb" / "750" / "hessianv4_"), qgrid=[2, 2, 2], Nqirr=4)


def read_matdyn_freq(file):
    """
    Read a ``flfrq`` file written by QuantumESPRESSO's matdyn.x.

    Returns the q-points in cartesian units of 2*pi/alat [nks, 3] and the
    frequencies in cm^-1 [nks, nbnd] (imaginary ones as negative values).
    """
    text = Path(file).read_text()
    header, body = text.split("/", 1)
    nbnd = int(re.search(r"nbnd\s*=\s*(\d+)", header).group(1))
    nks = int(re.search(r"nks\s*=\s*(\d+)", header).group(1))
    values = np.array(body.split(), dtype=float).reshape(nks, -1)
    return values[:, :3], values[:, -nbnd:]


def read_matdyn_path(file):
    """Read the high-symmetry points (fractional) and the points per segment from a band-form matdyn.in."""
    lines = Path(file).read_text().split("/", 1)[1].split("\n")
    lines = [line.split("!")[0].split() for line in lines if line.split("!")[0].strip()]
    npoints = int(lines[0][0])
    points = np.array([line[:3] for line in lines[1 : npoints + 1]], dtype=float)
    nsteps = [int(line[3]) for line in lines[1 : npoints + 1]]
    return points, nsteps
