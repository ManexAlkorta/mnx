"""
Generate ``cc_freq.txt``, the CellConstructor reference of ``tests/test_loto.py``.

CellConstructor interpolates the MnTe dyn files written by mnx (``DynMatrix.to_CC``) with
``ForceTensor.Tensor2``, which adds the long-range term and the non-analytic term at Gamma
with QuantumESPRESSO's rgd_blk and nonanal. Gamma takes the direction of its segment, as in
``DynMatrix.get_bands``. CellConstructor does not impose a charge sum rule, so mnx reads the
charges with ``zasr="no"``.

Run from the repository root:  python tests/data/MnTe_2x2x2/make_cc_reference.py
"""

from pathlib import Path

import numpy as np

import cellconstructor as CC
import cellconstructor.ForceTensor
import cellconstructor.Units

import mnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N = 20  # intervals per segment, as in matdyn.in
G, M, K, A, L, H = [0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0.5], [0.5, 0, 0.5], [1 / 3, 1 / 3, 0.5]
PATH = np.array([(G, M), (M, K), (K, G), (G, A), (A, L), (L, H), (H, A), (L, G)], dtype=float)


def main():
    dyn = mnx.DynMatrix.from_file(str(ROOT / "Tutorial" / "dyns" / "MnTe" / "dyn"), qgrid=[2, 2, 2], Nqirr=4, zasr="no")
    phonons = dyn.to_CC()
    tensor = CC.ForceTensor.Tensor2(
        phonons.structure, phonons.structure.generate_supercell(phonons.GetSupercell()), phonons.GetSupercell()
    )
    tensor.SetupFromPhonons(phonons)
    tensor.Center()

    bg = phonons.structure.get_reciprocal_vectors() / (2 * np.pi)  # cartesian q in 2pi/A
    masses = np.repeat(phonons.structure.get_masses_array(), 3)
    rows = []
    for k_start, k_end in PATH:
        for t in np.linspace(0, 1, N + 1):
            q = k_start + t * (k_end - k_start)
            # CellConstructor's Fourier phases have the opposite sign of QE's, hence -q.
            fc = tensor.Interpolate(-q @ bg, q_direct=(k_end - k_start) @ bg)
            eigenvalues = np.linalg.eigvalsh(fc / np.sqrt(np.outer(masses, masses)))
            frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * CC.Units.RY_TO_CM
            rows.append(np.concatenate([q, frequencies]))
    header = "CellConstructor Tensor2 frequencies of MnTe (cm^-1), zasr='no'. Columns: q (fractional, 3), 12 frequencies"
    np.savetxt(HERE / "cc_freq.txt", np.array(rows), fmt="%14.8f", header=header)


if __name__ == "__main__":
    main()
