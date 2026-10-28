"""
Phonon spectra on a q-grid: dynamical matrices, force constants and related tools.

The module exposes two classes:

* :class:`DynMatrix` holds the full phonon problem of a crystal: the structure, the
  dynamical matrices of every irreducible q-star of a grid and the real-space force
  constants obtained from them. It interpolates the spectrum to any q-point, builds
  band structures, distorts the structure along phonon modes and symmetrizes the
  force constants.
* :class:`DynQ` holds the dynamical matrices of a single q-star (one QuantumESPRESSO
  ``dyn`` file) together with their eigenvalues and eigenvectors.

Units follow QuantumESPRESSO unless stated otherwise: force constants in Ry/bohr^2,
masses in Ry atomic units, lengths in Angstrom and frequencies in cm^-1. Imaginary
frequencies are stored as complex numbers and reported as negative values in band
structures.

See ``Tutorial/DynMatrix.ipynb`` for a guided tour.
"""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import phonopy
import spglib

import cellconstructor as CC
import cellconstructor.Phonons

import mnx.symph as symph
import mnx.utils.cell as _cell
import mnx.utils.classify as _classify
import mnx.utils.consts as _consts
from mnx.FModules import interpolation as _interpolation

from .structure import Structure


class DynMatrix:
    """
    Phonon spectra of a crystal computed on a regular q-grid.

    Build instances with :meth:`from_file` (QuantumESPRESSO ``dyn`` files) or
    :meth:`from_phonopy` (VASP + phonopy). Usage examples are collected in the
    ``Tutorial/DynMatrix.ipynb`` notebook.

    Attributes
    ----------
    structure : Structure
        Primitive cell.
    super_structure : Structure
        Supercell commensurate with ``qgrid``, used for the real-space force constants.
    qgrid : list of int, [3]
        q-grid on which the dynamical matrices were computed.
    Nqirr : int
        Number of irreducible q-stars.
    Nqpoints : int
        Total number of q-points in the grid (sum over all stars).
    DynQs : np.ndarray of DynQ, [Nqirr]
        Dynamical matrices of each q-star.
    qstars : np.ndarray of np.ndarray, [Nqirr]
        q-points of each star in fractional units, ``qstars[s]`` has shape [Nqinstar, 3].
    phiR : np.ndarray, [Nat_sc, Nat_sc, 3, 3]
        Real-space force constants of the supercell.
    """

    @classmethod
    def from_file(cls, file: str, qgrid: list | None = None, Nqirr: int = 0) -> "DynMatrix":
        """
        Read the phonon spectra from a set of QuantumESPRESSO ``dyn`` files.

        Parameters
        ----------
        file : str
            Prefix of the dyn files, e.g. ``"./hessian"`` to read ``hessian1`` ...
            ``hessianN``. When ``Nqirr=0`` it is the full name of a single dyn file.
        qgrid : list of int, [3], optional
            q-grid on which the phonon spectra was computed. Defaults to ``[0, 0, 0]``,
            i.e. no grid (only meaningful when reading a single file).
        Nqirr : int, optional
            Number of irreducible q-points in the grid. With ``Nqirr=0`` (default) only
            the single file ``file`` is read and no real-space force constants are built.

        Returns
        -------
        DynMatrix
            The phonon spectra.

        Examples
        --------
        >>> dyn = DynMatrix.from_file("./dyns/CrSb/750/hessianv4_", qgrid=[2, 2, 2], Nqirr=4)
        """
        instance = cls()
        instance.qgrid = [0, 0, 0] if qgrid is None else qgrid
        files = [file] if Nqirr == 0 else [f"{file}{s + 1}" for s in range(Nqirr)]

        instance.structure = Structure.from_file(files[0], format="dyn")
        instance.Nqirr = len(files)
        instance.DynQs = np.empty(instance.Nqirr, dtype=object)
        instance.qstars = np.empty(instance.Nqirr, dtype=object)
        for s, name in enumerate(files):
            instance.DynQs[s] = DynQ.from_file(name)
            instance.qstars[s] = instance.DynQs[s].qpoints
        instance.Nqpoints = sum(dynq.Nqinstar for dynq in instance.DynQs)
        instance._alat = instance.DynQs[0]._alat

        if Nqirr > 0:
            instance.super_structure = instance.structure.expand_structure(instance.qgrid)
            instance.get_phiR()
        return instance

    @classmethod
    def from_phonopy(cls, folder: str, qgrid: list) -> "DynMatrix":
        """
        Read the force constants of a VASP + phonopy calculation.

        The structure is read from ``primPOSCAR`` (falling back to ``POSCAR``) and the
        force constants from ``phonopy_disp.yaml`` (falling back to ``phonopy.yaml``) inside
        ``folder``.

        Parameters
        ----------
        folder : str
            Folder in which the VASP + phonopy calculation was run.
        qgrid : list of int, [3]
            q-grid (supercell size) of the phonopy calculation.

        Returns
        -------
        DynMatrix
            The phonon spectra, with one single-point star per grid point.
        """
        instance = cls()
        instance.qgrid = qgrid
        instance.Nqirr = int(np.prod(qgrid))
        try:
            instance.structure = Structure.from_file(f"{folder}/primPOSCAR", format="vasp")
        except FileNotFoundError:
            instance.structure = Structure.from_file(f"{folder}/POSCAR", format="vasp")
        instance._alat = np.linalg.norm(instance.structure.cell[0, :])
        instance.super_structure = instance.structure.expand_structure(instance.qgrid)

        # phonopy looks for FORCE_SETS in the working directory unless told otherwise.
        force_sets = f"{folder}/FORCE_SETS" if os.path.isfile(f"{folder}/FORCE_SETS") else None
        try:
            phonon = phonopy.load(f"{folder}/phonopy_disp.yaml", force_sets_filename=force_sets)
        except FileNotFoundError:
            phonon = phonopy.load(f"{folder}/phonopy.yaml", force_sets_filename=force_sets)
        masses = phonon.masses / _consts.Ry2AMU
        instance.structure.masses = masses
        nat = instance.structure.Natoms

        qpoints = np.array([np.divide(idx, qgrid) for idx in np.ndindex(*qgrid)])
        instance.Nqpoints = len(qpoints)

        # phonopy gives D(q) in eV/A^2/amu with (atom, cart) superindices; convert it to
        # mass-weighted force constants Phi(q) with shape [nat, nat, 3, 3] in Ry/bohr^2.
        phiqs = np.empty([instance.Nqpoints, nat, nat, 3, 3], dtype=complex)
        mass_factor = np.sqrt(np.outer(masses, masses))[:, :, None, None]
        for qi, q in enumerate(qpoints):
            dynq = np.reshape(phonon.get_dynamical_matrix_at_q(q), [nat, 3, nat, 3])
            dynq = np.moveaxis(dynq * _consts.eVA2RyBohr2 * _consts.Ry2AMU, 1, 2)
            phiqs[qi] = dynq * mass_factor
        instance.phiR = np.real(
            _interpolation.ift_fcq(
                phiqs, qpoints, instance.structure.cell, instance.super_structure.atom_coords, "atomic"
            )
        )

        instance.DynQs = np.empty(instance.Nqirr, dtype=object)
        instance.qstars = np.empty(instance.Nqirr, dtype=object)
        for qi, q in enumerate(qpoints):
            instance.qstars[qi] = np.asarray([q])
            instance.DynQs[qi] = instance.get_dynq(q, gauge="lattice")
            instance.DynQs[qi]._diagonalize()  # same gauge-fixed, orthonormal modes as from_file
        return instance

    def distort_structure(self, Q: list, modes: list, mod: list | None = None) -> Structure:
        """
        Displace the atoms along a combination of phonon modes.

        Parameters
        ----------
        Q : list of array_like
            Amplitudes (order parameter) of each q-star taking part in the distortion,
            one entry per q-point of the star. Combining a one-dimensional and a
            three-dimensional star reads ``Q = [[Q1], [Q2, Q3, Q4]]``.
        modes : list of [int, int]
            ``[star, mode]`` pairs of the modes used for the distortion, in the same
            order as ``Q``. For the first mode of star 0 and the second mode of star 1:
            ``modes = [[0, 0], [1, 1]]``.
        mod : list of int, [3], optional
            Supercell of the distorted structure. Defaults to ``qgrid``. Every q-point
            of the selected stars must be commensurate with it.

        Returns
        -------
        Structure
            The distorted supercell. :meth:`project_structure` recovers ``Q`` from it.
        """
        if mod is None:
            mod = self.qgrid
        self._check_commensurate(modes, mod)

        supercell = self.structure.expand_structure(mod)
        atom_coords = supercell.atom_coords.copy()
        _, displacements, _ = self.expand_polvecs(modes=modes, mod=mod)
        for qs, displacements_instar in enumerate(displacements):
            for i, displacement in enumerate(displacements_instar):
                atom_coords += np.real(Q[qs][i] * np.reshape(displacement, atom_coords.shape))
        return Structure.from_data(supercell.cell, atom_coords, supercell.atom_species)

    def project_structure(self, structure: Structure, modes: list, cutoff: float = 0.2) -> list:
        r"""
        Decompose the displacements of a structure into phonon modes.

        This is the inverse of :meth:`distort_structure`: for ``s = distort_structure(Q,
        modes)``, ``project_structure(s, modes)`` returns ``Q``. The supercell is taken
        from ``structure.cell``, which must be a diagonal multiple of the primitive cell.
        Atoms are matched to the sites of the undistorted supercell by species and
        periodic distance, so their order in ``structure`` does not matter.

        The amplitudes solve, in the least-squares sense,

        .. math:: \sqrt{m_A}\, \Delta\mathbf r_A = \sum_{s,i} \mathrm{Re}\left[Q_{s,i}\, \mathbf e^{sc}_{s,i}(A)\right]

        over the selected modes, with the same normalized supercell vectors used by
        :meth:`distort_structure`. Modes are orthonormal, so components along modes that
        are not selected do not leak into the result. At self-conjugate q-points (Gamma
        and zone-boundary points such as M or L) the modes are real and only Re(Q) has an
        effect, so Q comes back real. When q and -q are both in a star, a distortion only
        fixes Q_q + conj(Q_-q); the result splits it evenly, Q_-q = conj(Q_q).

        Parameters
        ----------
        structure : Structure
            Distorted supercell, in any atom order.
        modes : list of [int, int]
            ``[star, mode]`` pairs to project onto, as in :meth:`distort_structure`.
        cutoff : float, optional
            Largest displacement in Angstrom allowed when matching each atom to its site
            in the undistorted supercell. Defaults to 0.2.

        Returns
        -------
        list of np.ndarray
            For each entry of ``modes``, the complex amplitudes of every q-point of its
            star, in the units of ``Q`` in :meth:`distort_structure`.

        Raises
        ------
        ValueError
            If the cell is not a diagonal supercell, a q-point is not commensurate with
            it, or the atoms cannot be matched within ``cutoff``.

        Examples
        --------
        >>> distorted = dyn.distort_structure(Q=[[10, 0, 0], [0, 10, 10]], modes=[[2, 0], [3, 0]])
        >>> dyn.project_structure(distorted, modes=[[2, 0], [3, 0]])
        [array([10.+0.j, 0.+0.j, 0.+0.j]), array([0.+0.j, 10.+0.j, 10.+0.j])]
        """
        mod = self._supercell_of(structure)
        self._check_commensurate(modes, mod)
        reference = self.structure.expand_structure(mod)
        _, displacements = _cell.match_atoms(structure, reference, cutoff)

        polvecs, _, exp_masses = self.expand_polvecs(modes, mod)
        target = np.sqrt(exp_masses) * displacements.ravel()
        # Re[Q e] = Re(Q) Re(e) - Im(Q) Im(e): two real columns per q-point and mode.
        columns = [part for polvecs_instar in polvecs for e in polvecs_instar for part in (e.real, -e.imag)]
        coefficients = np.linalg.lstsq(np.column_stack(columns), target, rcond=1e-10)[0]
        Q = coefficients[0::2] + 1j * coefficients[1::2]
        return np.split(Q, np.cumsum([len(polvecs_instar) for polvecs_instar in polvecs])[:-1])

    def info(self) -> None:
        """Print a summary of the structure and of the q-stars."""
        self.structure.info()
        print("Dynamical information:")
        print(f"\tNumber of stars: {self.Nqirr}")
        for qs in range(self.Nqirr):
            for qi, q in enumerate(self.DynQs[qs].qpoints):
                if qi == 0:
                    print("\t\tStar {0:>3}: [{1:14.8f},{2:14.8f},{3:14.8f}]".format(qs, *q))
                else:
                    print("\t\t          [{0:14.8f},{1:14.8f},{2:14.8f}]".format(*q))
            if qs != self.Nqirr - 1:
                print("\n")

    def expand_polvecs(self, modes: list, mod: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand polarization vectors from the primitive cell to a supercell.

        Parameters
        ----------
        modes : list of [int, int]
            ``[star, mode]`` pairs to expand, e.g. ``[[2, 0]]`` for mode 0 of star 2.
        mod : list of int, [3]
            Supercell over which the modes are expanded, typically ``qgrid``.

        Returns
        -------
        polvecs_list : np.ndarray of np.ndarray, [len(modes)]
            For each requested mode, the normalized polarization vectors of every
            q-point of its star, shape [Nqinstar, Nat_sc*3].
        displacements_list : np.ndarray of np.ndarray, [len(modes)]
            Same as ``polvecs_list`` divided by the square root of the atomic masses.
        exp_masses : np.ndarray, [Nat_sc*3]
            Mass associated with each component of the supercell vectors.
        """
        polvecs_list = np.empty(len(modes), dtype=object)
        displacements_list = np.empty(len(modes), dtype=object)
        for ii, (star, mode) in enumerate(modes):
            polvecs, displacements, _ = self.DynQs[star]._expand_polvecs([mode], mod)
            polvecs_list[ii], displacements_list[ii] = polvecs[0], displacements[0]
        return polvecs_list, displacements_list, _expanded_masses(self.structure.masses, mod)

    def get_phiR(self) -> None:
        """
        Fourier transform the force constants of all q-points to real space.

        Stores the result in ``self.phiR`` with shape [Nat_sc, Nat_sc, 3, 3]. The
        transform is done by the Fortran routine ``FModules.interpolation.ift_fcq``.
        """
        phiqs = np.concatenate([dynq.phiqs for dynq in self.DynQs])
        qpoints = np.concatenate([dynq.qpoints for dynq in self.DynQs])
        self.phiR = np.real(
            _interpolation.ift_fcq(phiqs, qpoints, self.structure.cell, self.super_structure.atom_coords, "lattice")
        )

    def get_dynq(self, q: list, gauge: str = "atomic") -> "DynQ":
        r"""
        Interpolate the force constants :math:`\Phi(q)` to an arbitrary q-point.

        Parameters
        ----------
        q : array_like, [3]
            q-point in fractional units.
        gauge : {"atomic", "lattice"}, optional
            Gauge of the Fourier interpolation. Defaults to ``"atomic"``.

        Returns
        -------
        DynQ
            Single-point star with the interpolated spectrum.
        """
        phiq, eigenvalues, polvecs = self._interpolate(q, gauge)
        return DynQ.from_data(self.structure, np.array([q]), np.array([phiq]), np.array([eigenvalues]), np.array([polvecs]))

    def get_bands(self, k_inpath: np.ndarray, N: int, gauge: str = "atomic") -> tuple[np.ndarray, np.ndarray, list]:
        """
        Compute the phonon dispersion along a path in reciprocal space.

        Parameters
        ----------
        k_inpath : array_like, [Nsegments, 2, 3]
            Start and end points of each path segment in fractional units, e.g.
            ``[[[0, 0, 0], [0.5, 0, 0]], [[0.5, 0, 0], [0.5, 0, 0.5]]]`` for
            Gamma-M-A in a hexagonal cell.
        N : int
            Number of intervals per segment (each segment has ``N + 1`` points).
        gauge : {"atomic", "lattice"}, optional
            Gauge of the Fourier interpolation. Defaults to ``"atomic"``.

        Returns
        -------
        bands : np.ndarray, [Nsegments*(N+1), Natoms*3]
            Frequencies in cm^-1, imaginary ones shown as negative.
        cart_qpath : np.ndarray, [Nsegments*(N+1)]
            Accumulated cartesian length along the path, for plotting.
        xticks : list of float
            Values of ``cart_qpath`` at the segment boundaries.
        """
        qpath, cart_qpath, xticks = self._build_qpath(k_inpath, N)
        bands = np.empty([len(qpath), self.structure.Natoms * 3], dtype=np.float64)
        for qi, q in enumerate(qpath):
            _, eigenvalues, _ = self._interpolate(q, gauge)
            bands[qi] = _signed_frequencies(_eigenvalues_to_cm(eigenvalues))
        return bands, cart_qpath, xticks

    def get_bands_with_SF(self, k_inpath: np.ndarray, N: int, Q: list) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """
        Compute the phonon dispersion and the one-phonon structure factor along a path.

        The structure factor of each mode is evaluated at the momentum ``Q + k``, with
        ``k`` running along the path and the atomic gauge for the interpolation.

        Parameters
        ----------
        k_inpath : array_like, [Nsegments, 2, 3]
            Start and end points of each path segment in fractional units.
        N : int
            Number of intervals per segment (each segment has ``N + 1`` points).
        Q : array_like, [3]
            Reciprocal lattice vector in fractional units, e.g. ``[0, 0, 3]``.

        Returns
        -------
        bands : np.ndarray, [Nsegments*(N+1), Natoms*3]
            Frequencies in cm^-1, imaginary ones shown as negative.
        cart_qpath : np.ndarray, [Nsegments*(N+1)]
            Accumulated cartesian length along the path, for plotting.
        xticks : list of float
            Values of ``cart_qpath`` at the segment boundaries.
        structure_factor : np.ndarray, [Nsegments*(N+1), Natoms*3]
            Structure factor of each mode.
        """
        qpath, cart_qpath, xticks = self._build_qpath(k_inpath, N)
        nmodes = self.structure.Natoms * 3
        masses = self.structure.masses
        bands = np.empty([len(qpath), nmodes], dtype=np.float64)
        structure_factor = np.empty([len(qpath), nmodes], dtype=np.float64)
        for qi, q in enumerate(qpath):
            _, eigenvalues, polvecs = self._interpolate(q, "atomic")
            frequencies = _eigenvalues_to_cm(eigenvalues)
            bands[qi] = _signed_frequencies(frequencies)

            displacements = np.reshape(polvecs / np.sqrt(_expanded_masses(masses)), [nmodes, -1, 3])
            Qq_cart = _cell.cryst2cart(np.asarray(q + Q), self.structure.rcell)
            phases = np.exp(2j * np.pi * (self.structure.atom_coords @ Qq_cart))
            weights = (phases * np.sqrt(2 * masses))[None, :, None]
            s = np.einsum("man,n->m", weights * displacements, Qq_cart)
            structure_factor[qi] = np.real(np.conjugate(s) * s / frequencies)
        return bands, cart_qpath, xticks, structure_factor

    def Symmetrize(self, symprec: float = 1e-5, apply_translations: bool = True) -> None:
        """
        Symmetrize the crystal structure and the force constants.

        The structure is symmetrized with spglib, and the real-space force constants with
        QuantumESPRESSO's routines (permutation, pure translations and space group). The
        dynamical matrices of every q-star are then recomputed from them.

        Parameters
        ----------
        symprec : float, optional
            Tolerance used by spglib to find the symmetry operations; see
            ``Structure.get_spacegroup()`` to choose the target space group.
            Defaults to 1e-5.
        apply_translations : bool, optional
            Whether to impose invariance under the pure translations of the supercell.
            Defaults to True.
        """
        self.structure.Symmetrize(symprec)
        self.super_structure = self.structure.expand_structure(self.qgrid)
        self.SetupFromSPGLIB(symprec=symprec)

        # Permutation symmetry Phi_ab^ij = Phi_ba^ji.
        phiR = 0.5 * (self.phiR + self.phiR.transpose(1, 0, 3, 2))

        # The Fortran routines work in place on arrays shaped (3, 3, nat, nat).
        fc = np.array(phiR.transpose(2, 3, 0, 1), dtype=np.double, order="F")
        if apply_translations:
            assert np.ndim(self._QE_translations_irt) == 2, "Error, symmetries not set up to work in the supercell"
            symph.trans_v2(fc, self._QE_translations_irt)
        symph.sym_v2(fc, self._QE_at, self._QE_bg, self._QE_s, self._QE_irt, self._QE_nsym)
        self.phiR[...] = fc.transpose(2, 3, 0, 1)

        for si, star in enumerate(self.qstars):
            dynq = self.DynQs[si]
            for qi, q in enumerate(star):
                dynq.phiqs[qi] = self._interpolate(q, "lattice")[0]
            dynq.dynqs = dynq._phis2dyns()
            dynq._diagonalize()

    def SetupFromSPGLIB(self, symprec: float) -> None:
        """
        Prepare the symmetry data needed by QuantumESPRESSO's symmetrization routines.

        Fills the auxiliary ``self._QE_*`` attributes (rotations, atom mappings, pure
        translations...) from the spglib symmetries of the supercell.

        Parameters
        ----------
        symprec : float
            Tolerance used by spglib to find the symmetry operations.
        """
        spg_syms = spglib.get_symmetry(self.super_structure.get_spglib_cell(), symprec)
        symmetries = _classify.GetSymmetriesFromSPGLIB(spg_syms)
        nat = self.super_structure.Natoms

        self._QE_nat = nat
        self._QE_s = np.zeros((3, 3, 48), dtype=np.intc, order="F")
        self._QE_s_cart = np.zeros((3, 3, 48), dtype=np.float64, order="F")
        self._QE_s_inv_cart = np.zeros((3, 3, 48), dtype=np.float64, order="F")
        self._QE_irt = np.zeros((48, nat), dtype=np.intc, order="F")
        self._QE_invs = np.zeros(48, dtype=np.intc, order="F")
        self._QE_at = np.asfortranarray(self.super_structure.cell.T, dtype=np.float64)
        self._QE_bg = np.asfortranarray(self.super_structure.rcell.T, dtype=np.float64)

        # spglib lists the point-group operations once per pure translation; the
        # point group ends where the first rotation appears again.
        n_syms = 0
        for i, sym in enumerate(symmetries):
            rot = sym[:, :3]
            if i > 0 and np.sum((rot - symmetries[0][:, :3]) ** 2) < 0.1:
                n_syms = i
                break
            rot_cart = _classify.convert_matrix_cart_cryst2(rot, self.super_structure.cell, cryst_to_cart=True)
            self._QE_s[:, :, i] = rot.T
            self._QE_s_cart[:, :, i] = rot_cart
            self._QE_s_inv_cart[:, :, i] = rot_cart.T
            self._QE_irt[i, :] = _classify.GetIRT(self.super_structure, sym) + 1  # Fortran indexing
        if n_syms == 0:
            n_syms = len(symmetries)

        self._QE_translation_nr = len(symmetries) // n_syms
        self._QE_nsymq = n_syms
        self._QE_nsym = n_syms

        self._QE_translations_irt = np.zeros((nat, self.Nqpoints), dtype=np.intc, order="F")
        self._QE_translations = np.zeros((3, self.Nqpoints), dtype=np.double, order="F")
        for i in range(self.Nqpoints):
            sym = symmetries[i * n_syms]
            assert np.sum((np.eye(3) - sym[:, :3]) ** 2) < 0.5, (
                "Error, symmetries are not correctly ordered. They must always start with the identity.\n"
                f"N_syms = {n_syms}; N = {i * n_syms}; SYM = {sym}"
            )
            self._QE_translations_irt[:, i] = _classify.GetIRT(self.super_structure, sym) + 1
            self._QE_translations[:, i] = sym[:, 3]

        self._QE_invs[:] = _classify.get_invs(self._QE_s, self._QE_nsym)

    def change_cell(self, cell: np.ndarray) -> None:
        """
        Replace the unit cell, keeping the fractional atomic coordinates fixed.

        Parameters
        ----------
        cell : np.ndarray, [3, 3]
            New unit cell in Angstrom, one lattice vector per row.
        """
        self.structure.atom_coords = _cell.cryst2cart(_cell.cart2cryst(self.structure.atom_coords, self.structure.cell), cell)
        self.structure.cell = cell
        self.structure.rcell = _cell.get_rcell(self.structure.cell)
        self._alat = np.linalg.norm(self.structure.cell[0, :]) / _consts.bohr2angstroms
        for dynq in self.DynQs:
            dynq.structure = self.structure
            dynq._alat = self._alat

    def copy(self) -> "DynMatrix":
        """Return a deep copy of the object."""
        return copy.deepcopy(self)

    def write(self, file: str) -> None:
        """
        Write every q-star in QuantumESPRESSO format as ``{file}1`` ... ``{file}Nqirr``.

        Parameters
        ----------
        file : str
            Prefix of the output files.
        """
        for si, dynq in enumerate(self.DynQs):
            dynq.write(f"{file}{si + 1}")

    def to_CC(self) -> "CC.Phonons.Phonons":
        """
        Convert to a CellConstructor ``Phonons`` object.

        The dyn files are written to a temporary folder and read back with CellConstructor.
        """
        with tempfile.TemporaryDirectory() as tmp:
            prefix = f"{tmp}/dyn"
            self.write(prefix)
            return CC.Phonons.Phonons(prefix, nqirr=self.Nqirr)

    def get_Pmn(self, symprec: float = 1e-5) -> None:
        r"""
        Build the representation matrices of the little group in the phonon basis.

        For every q-star stores ``DynQ.Pmn`` with
        :math:`P_{\mu\nu}(S) = \langle e_\mu | S | e_\nu \rangle` for the operations ``S``
        of the little group of q, restricted to pairs of degenerate modes (other
        entries are zero). Use :meth:`DynQ.compute_traces` afterwards to get the characters.

        Parameters
        ----------
        symprec : float, optional
            Tolerance used to find the symmetry operations. Defaults to 1e-5.
        """
        mapping, _, rot_cart = _cell.map_singlet(self.to_CC(), symprec=symprec)

        nmodes = self.structure.Natoms * 3
        nat_sc = self.super_structure.Natoms
        for dynq in self.DynQs:
            expanded = dynq._expand_polvecs(range(nmodes), self.qgrid)[0]
            polvecs = np.array([expanded[mu][0] for mu in range(nmodes)])  # first q of the star
            polvecs_atoms = np.reshape(polvecs, [nmodes, nat_sc, 3])
            dynq.Pmn = np.zeros([nmodes, nmodes, len(dynq.lg_idx)], dtype=np.complex128)
            for li, isym in enumerate(dynq.lg_idx):
                rotated = np.zeros_like(polvecs_atoms)
                rotated[:, mapping[:, isym], :] = polvecs_atoms @ rot_cart[isym].T
                P = np.conjugate(polvecs) @ np.reshape(rotated, [nmodes, -1]).T
                dynq.Pmn[:, :, li] = np.where(dynq.degs, P, 0)

    def _supercell_of(self, structure: Structure) -> list:
        """Diagonal supercell ``mod`` such that ``structure.cell`` = mod x primitive cell."""
        M = structure.cell @ np.linalg.inv(self.structure.cell)
        mod = np.round(np.diag(M)).astype(int)
        if np.any(mod < 1) or not np.allclose(M, np.diag(mod), atol=1e-4):
            raise ValueError(
                "The cell of the structure is not a diagonal supercell of the primitive cell; "
                f"structure.cell @ inv(primitive cell) =\n{np.round(M, 4)}"
            )
        return mod.tolist()

    def _check_commensurate(self, modes: list, mod: list) -> None:
        """Raise if a q-point of the selected stars is not periodic in the ``mod`` supercell."""
        for star, _ in modes:
            phases = self.DynQs[star].qpoints * np.asarray(mod)
            if not np.allclose(phases, np.round(phases), atol=1e-6):
                raise ValueError(f"Star {star} has q-points that are not commensurate with the supercell {list(mod)}")

    def _interpolate(self, q: np.ndarray, gauge: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fourier interpolate the force constants to ``q`` and diagonalize them.

        Returns the force constants [Natoms, Natoms, 3, 3], the eigenvalues of the
        dynamical matrix [Natoms*3] (Ry units, ascending) and the eigenvectors as rows
        [Natoms*3, Natoms*3].
        """
        phiq, eigenvalues, polvecs = _interpolation.ft_fcr(
            self.phiR, q, self.structure.masses, self.qgrid, self.structure.cell, self.super_structure.atom_coords, gauge
        )
        return phiq, eigenvalues, np.transpose(polvecs)

    def _build_qpath(self, k_inpath: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Sample the segments of ``k_inpath`` with ``N + 1`` points each.

        Returns the fractional q-points [Nsegments*(N+1), 3], the accumulated cartesian
        path length (reciprocal cell without 2*pi, scaled by 10, i.e. nm^-1) and the
        path length at the segment boundaries.
        """
        k_inpath = np.asarray(k_inpath, dtype=float)
        t = np.linspace(0, 1, N + 1)
        qpath = np.empty([len(k_inpath) * (N + 1), 3])
        cart_qpath = np.zeros(len(k_inpath) * (N + 1))
        xticks = [0]
        for ki, (k_start, k_end) in enumerate(k_inpath):
            segment = slice(ki * (N + 1), (ki + 1) * (N + 1))
            dk = k_end - k_start
            offset = cart_qpath[ki * (N + 1) - 1] if ki > 0 else 0
            cart_qpath[segment] = offset + np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell)) * 10 * t
            xticks.append(cart_qpath[segment][-1])
            qpath[segment] = k_start + t[:, None] * dk[None, :]
        return qpath, cart_qpath, xticks


class DynQ:
    """
    Dynamical matrices of a single q-star.

    Build instances with :meth:`from_file` (one QuantumESPRESSO dyn file) or
    :meth:`from_data`.

    Attributes
    ----------
    structure : Structure
        Primitive cell.
    Nqinstar : int
        Number of q-points in the star.
    qpoints : np.ndarray, [Nqinstar, 3]
        q-points in fractional units.
    qpoints_cart : np.ndarray, [Nqinstar, 3]
        q-points in cartesian units (Angstrom^-1, without 2*pi).
    phiqs : np.ndarray, [Nqinstar, Natoms, Natoms, 3, 3]
        Force constants at each q-point (Ry/bohr^2).
    dynqs : np.ndarray, [Nqinstar, Natoms, Natoms, 3, 3]
        Mass-scaled force constants, i.e. the dynamical matrices.
    frequencies : np.ndarray, [Nqinstar, Natoms*3], complex
        Phonon frequencies in cm^-1, sorted in ascending order of the eigenvalue.
        Imaginary frequencies have a vanishing real part.
    polvecs : np.ndarray, [Nqinstar, Natoms*3, Natoms*3]
        Polarization vectors (eigenvectors of the dynamical matrix), indexed as
        [q-point, mode, 3*atom + cartesian].
    displacements : np.ndarray, [Nqinstar, Natoms*3, Natoms*3]
        Polarization vectors divided by the square root of the atomic masses.
    lg_idx : list of int
        Indices of the spglib symmetries of the little group of the star.
    degs : np.ndarray, [Natoms*3, Natoms*3], bool
        ``degs[mu, nu]`` is True when modes ``mu`` and ``nu`` are degenerate.
    """

    @classmethod
    def from_file(cls, file: str) -> "DynQ":
        """
        Read the dynamical matrices of a q-star from a QuantumESPRESSO dyn file.

        Parameters
        ----------
        file : str
            Name of the dyn file.

        Returns
        -------
        DynQ
            The q-star, diagonalized.
        """
        instance = cls()
        instance.structure = Structure.from_file(file, format="dyn")
        nat = instance.structure.Natoms

        with open(file, "r") as f:
            lines = [line.split() for line in f]
        instance._alat = float(lines[2][3]) * _consts.bohr2angstroms

        # Each q-point starts with a "q = ( qx qy qz )" line followed by nat*nat blocks
        # of an "i j" header and 3 rows of (re, im) pairs. The last "q" line belongs
        # to the diagonalization section and is skipped.
        tokens = [line for line in lines if line]
        q_rows = [i for i, line in enumerate(tokens) if line[0] == "q"][:-1]
        instance.Nqinstar = len(q_rows)
        instance.qpoints_cart = np.array([tokens[i][3:6] for i in q_rows], dtype=float) / instance._alat
        instance.qpoints = _cell.cart2cryst(instance.qpoints_cart, instance.structure.rcell)
        instance.phiqs = np.empty([instance.Nqinstar, nat, nat, 3, 3], dtype=complex)
        for qi, row in enumerate(q_rows):
            block = tokens[row + 1 : row + 1 + 4 * nat * nat]
            values = np.array([line for k, line in enumerate(block) if k % 4], dtype=np.float64)
            instance.phiqs[qi] = np.reshape(values.view(np.complex128), [nat, nat, 3, 3])  # (re, im) pairs

        instance.dynqs = instance._phis2dyns()
        instance._diagonalize()
        instance.lg_idx = _cell.get_lg_idx(instance.structure, instance.qpoints[0])  # Same for the whole star.
        return instance

    @classmethod
    def from_data(cls, structure: Structure, qpoints: np.ndarray, phiqs: np.ndarray, frequencies: np.ndarray, polvecs: np.ndarray) -> "DynQ":
        """
        Build a q-star from already diagonalized data.

        Parameters
        ----------
        structure : Structure
            Primitive cell.
        qpoints : np.ndarray, [Nqinstar, 3]
            q-points of the star in fractional units.
        phiqs : np.ndarray, [Nqinstar, Natoms, Natoms, 3, 3]
            Force constants at each q-point (Ry/bohr^2).
        frequencies : np.ndarray, [Nqinstar, Natoms*3]
            Eigenvalues of the dynamical matrix (squared frequencies, Ry units), as
            returned by ``FModules.interpolation.ft_fcr``. They are converted to cm^-1.
        polvecs : np.ndarray, [Nqinstar, Natoms*3, Natoms*3]
            Eigenvectors of the dynamical matrix, indexed as
            [q-point, mode, 3*atom + cartesian].

        Returns
        -------
        DynQ
            The q-star.
        """
        instance = cls()
        instance.structure = structure
        instance._alat = np.linalg.norm(instance.structure.cell[0, :])
        instance.qpoints = qpoints
        instance.Nqinstar = qpoints.shape[0]
        instance.qpoints_cart = np.atleast_2d(_cell.cryst2cart(qpoints, structure.rcell))
        instance.phiqs = phiqs
        instance.dynqs = instance._phis2dyns()
        instance.frequencies = np.sqrt(np.asarray(frequencies, dtype=np.complex128)) * _consts.Ry2cm
        instance.polvecs = np.asarray(polvecs, dtype=np.complex128)
        instance.displacements = instance.polvecs / np.sqrt(_expanded_masses(structure.masses))

        instance.lg_idx = _cell.get_lg_idx(instance.structure, instance.qpoints[0])  # Same for the whole star.
        instance._find_degs()
        return instance

    def _phis2dyns(self) -> np.ndarray:
        """Return the dynamical matrices, i.e. ``phiqs`` divided by sqrt(m_a m_b)."""
        masses = self.structure.masses
        return self.phiqs / np.sqrt(np.outer(masses, masses))[None, :, :, None, None]

    def _diagonalize(self) -> None:
        """
        Diagonalize the dynamical matrices of the star with orthonormal eigenvectors.

        Sets ``frequencies``, ``polvecs``, ``displacements`` and ``degs`` from ``dynqs``
        (lattice gauge). The dynamical matrix is Hermitian, so ``np.linalg.eigh`` gives
        orthonormal modes, also inside degenerate subspaces. Two gauge choices make the
        modes reproducible and real displacements easy to expand:

        * at self-conjugate q-points (2q = G, e.g. Gamma, M, L) D(q) is real and the
          eigenvectors are chosen real;
        * when -q (mod G) is also in the star, its eigenvectors are taken as the complex
          conjugate of those at q, as allowed by D(-q) = D(q)*.
        """
        nmodes = self.structure.Natoms * 3
        matrices = self.dynqs.transpose(0, 1, 3, 2, 4).reshape(self.Nqinstar, nmodes, nmodes)
        eigenvalues = np.empty([self.Nqinstar, nmodes])
        self.polvecs = np.empty([self.Nqinstar, nmodes, nmodes], dtype=np.complex128)
        for qi, q in enumerate(self.qpoints):
            D = 0.5 * (matrices[qi] + matrices[qi].conj().T)
            partner = self._time_reversal_partner(qi)
            if partner is not None and np.allclose(D, matrices[partner].conj(), atol=1e-10 * np.abs(D).max()):
                eigenvalues[qi], self.polvecs[qi] = eigenvalues[partner], self.polvecs[partner].conj()
                continue
            if _is_self_conjugate(q) and np.abs(D.imag).max() <= 1e-10 * np.abs(D).max():
                D = D.real
            values, vectors = np.linalg.eigh(D)
            eigenvalues[qi], self.polvecs[qi] = values, vectors.T
        self.frequencies = _eigenvalues_to_cm(eigenvalues)
        self.displacements = self.polvecs / np.sqrt(_expanded_masses(self.structure.masses))
        self._find_degs()

    def _time_reversal_partner(self, qi: int) -> int | None:
        """Index of an earlier q-point of the star equal to -q[qi] modulo a reciprocal lattice vector."""
        for qj in range(qi):
            if _is_self_conjugate(0.5 * (self.qpoints[qi] + self.qpoints[qj])):
                return qj
        return None

    def copy(self) -> "DynQ":
        """Return a deep copy of the object."""
        return copy.deepcopy(self)

    def write(self, file: str, alat: bool = True) -> None:
        """
        Write the q-star in QuantumESPRESSO dyn format.

        Parameters
        ----------
        file : str
            Name of the output file.
        alat : bool, optional
            Write lengths in units of alat, as QuantumESPRESSO does. Needed if the file
            is to be used in a later QuantumESPRESSO calculation. Defaults to True.
        """
        fmt_general_info = "{0:>3d}{1:>5d}{2:>4d}{3:14.7f}{4:14.7f}{5:14.7f}{6:14.7f}{7:14.7f}{8:14.7f}\n"
        fmt_cell = "  {0:15.9f}{1:15.9f}{2:15.9f}\n"
        fmt_masses = " {0:>12d}  '{1:<4}'{2:20.9f}\n"
        fmt_atomic_position = "{0:>5d}{1:>5d} {2:16.10f} {3:16.10f} {4:16.10f}\n"
        fmt_dyn_matrix = "{0:12.8f} {1:12.8f}   {2:12.8f} {3:12.8f}   {4:12.8f} {5:12.8f}\n"
        fmt_q = "     q = ( {0:14.9f}{1:14.9f}{2:14.9f} )\n\n"
        fmt_freq = "     freq ({0:>4d}) = {1:14.6f} [THz] = {2:14.6f} [cm-1]\n"
        fmt_mode = "({0:10.6f}{1:10.6f}{2:10.6f}{3:10.6f}{4:10.6f}{5:10.6f} )\n"

        structure = self.structure
        nat = structure.Natoms
        norm = np.linalg.norm(structure.cell[0, :]) if alat else 1

        # One (label, mass) per species, in order of first appearance.
        species = {}
        for atom in range(nat):
            species.setdefault(structure.atom_species[atom, 0], structure.masses[atom])

        with open(file, "w") as f:
            f.write("Dynamical matrix file\n\n")
            f.write(fmt_general_info.format(structure.Nspecies, nat, 0, norm * _consts.angstroms2bohr, 0, 0, 0, 0, 0))
            f.write("Basis vectors\n")
            for vector in structure.cell:
                f.write(fmt_cell.format(*(vector / norm)))
            for s, (name, mass) in enumerate(species.items()):
                f.write(fmt_masses.format(s + 1, name, mass))
            for atom in range(nat):
                f.write(fmt_atomic_position.format(atom + 1, int(structure.atom_species[atom, 1]), *(structure.atom_coords[atom] / norm)))

            for qi in range(self.Nqinstar):
                f.write("\n     Dynamical  Matrix in cartesian axes\n\n")
                f.write(fmt_q.format(*(self.qpoints_cart[qi] * norm)))
                for i in range(nat):
                    for j in range(nat):
                        f.write("{0:>5d}{1:>5d}\n".format(i + 1, j + 1))
                        for row in self.phiqs[qi, i, j]:
                            f.write(fmt_dyn_matrix.format(*np.column_stack([row.real, row.imag]).ravel()))

            f.write("\n     Diagonalizing the dynamical matrix\n\n")
            f.write(fmt_q.format(*(self.qpoints_cart[0] * norm)))
            f.write(" **************************************************************************\n")
            for mode in range(nat * 3):
                frequency = self.frequencies[0, mode]
                if np.square(frequency).real < 0:
                    frequency_cm = -np.abs(frequency.imag)
                else:
                    frequency_cm = frequency.real
                f.write(fmt_freq.format(mode + 1, frequency_cm * _consts.cm2Thz, frequency_cm))
                disp = self.displacements[0, mode, :] / np.linalg.norm(self.displacements[0, mode, :])
                for atom in range(nat):
                    f.write(fmt_mode.format(*np.column_stack([disp[3 * atom : 3 * atom + 3].real, disp[3 * atom : 3 * atom + 3].imag]).ravel()))
            f.write(" **************************************************************************\n")

    def _find_degs(self, tol: float = 1e-3) -> None:
        """
        Find degenerate modes of the first q-point of the star.

        Sets ``self.degs[mu, nu]`` to True when the frequencies of ``mu`` and ``nu``
        differ by less than ``tol`` (cm^-1).
        """
        frequencies = self.frequencies[0]
        self.degs = np.abs(frequencies[:, None] - frequencies[None, :]) < tol

    def _expand_polvecs(self, modes: list, mod: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand polarization vectors from the primitive cell to a supercell.

        The vector of the cell at lattice position R = (i, j, k) is the primitive one
        multiplied by exp(2*pi*i q.R); cells are ordered with k running fastest.

        Parameters
        ----------
        modes : list of int
            Modes to expand, e.g. ``[0, 2, 3]``.
        mod : list of int, [3]
            Supercell over which the modes are expanded, typically the q-grid.

        Returns
        -------
        polvecs_list : np.ndarray of np.ndarray, [len(modes)]
            For each mode, the normalized polarization vectors of every q-point of the
            star, shape [Nqinstar, Nat_sc*3].
        displacements_list : np.ndarray of np.ndarray, [len(modes)]
            Same as ``polvecs_list`` divided by the square root of the atomic masses.
        exp_masses : np.ndarray, [Nat_sc*3]
            Mass associated with each component of the supercell vectors.
        """
        exp_masses = _expanded_masses(self.structure.masses, mod)
        lattice_points = np.array(list(np.ndindex(*mod)))
        # q-points read from files carry round-off (e.g. -0.49999999); snap the
        # commensurate ones to the exact grid so that e.g. zone-boundary phases are +-1.
        scaled = self.qpoints * np.asarray(mod)
        qpoints = np.where(np.abs(scaled - np.round(scaled)) < 1e-6, np.round(scaled) / np.asarray(mod), self.qpoints)
        phases = np.exp(2j * np.pi * (qpoints @ lattice_points.T))  # [Nqinstar, Ncells]

        polvecs_list = np.empty(len(modes), dtype=object)
        displacements_list = np.empty(len(modes), dtype=object)
        for mi, mode in enumerate(modes):
            polvecs = np.reshape(phases[:, :, None] * self.polvecs[:, mode, None, :], [self.Nqinstar, -1])
            polvecs = polvecs / np.linalg.norm(polvecs, axis=1, keepdims=True)
            polvecs_list[mi] = polvecs
            displacements_list[mi] = polvecs / np.sqrt(exp_masses)
        return polvecs_list, displacements_list, exp_masses

    def compute_traces(self) -> None:
        """
        Compute the characters of the little-group representations.

        Requires ``self.Pmn`` (see :meth:`DynMatrix.get_Pmn`). Sets ``self.traces``,
        shape [Natoms*3, len(lg_idx)], where each mode gets the trace of ``Pmn`` over
        its block of degenerate modes.
        """
        diagonal = np.einsum("kkl->kl", self.Pmn)
        self.traces = self.degs.astype(np.complex128) @ diagonal


def _expanded_masses(masses: np.ndarray, mod: list = (1, 1, 1)) -> np.ndarray:
    """Masses repeated for each cartesian component and each cell of the ``mod`` supercell."""
    return np.tile(np.repeat(masses, 3), int(np.prod(mod)))


def _is_self_conjugate(q: np.ndarray, tol: float = 1e-6) -> bool:
    """True when -q equals q modulo a reciprocal lattice vector (q in fractional units)."""
    return bool(np.allclose(2 * np.asarray(q), np.round(2 * np.asarray(q)), atol=tol))


def _eigenvalues_to_cm(eigenvalues: np.ndarray) -> np.ndarray:
    """Convert eigenvalues of the dynamical matrix (Ry units) to complex frequencies in cm^-1."""
    return np.sqrt(np.asarray(eigenvalues, dtype=np.complex128)) * _consts.Ry2cm


def _signed_frequencies(frequencies: np.ndarray) -> np.ndarray:
    """
    Real frequencies for plotting, with the leading imaginary ones shown as negative.

    ``frequencies`` must be sorted by eigenvalue, so imaginary modes come first.
    """
    imaginary = np.cumprod(np.square(frequencies).real < 0).astype(bool)
    return np.where(imaginary, -np.sqrt(np.abs(np.square(frequencies))), frequencies.real)
