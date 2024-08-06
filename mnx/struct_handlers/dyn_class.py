import numpy as np
import ase.data

import mnx.utils.cell_utils as cu
from mnx.struct_handlers.struct_class import STRUCT
import utils.consts as consts


class DYN_MATRIX:
    """DYNAMICAL MATRIX OBJECT. Reads a dynamical matrix at Quantum
    Espresso format, and handles the data in it.

        self.polarization_vectors: Eigensolutions of the dynamical matrix.
            np.array. [Nqpoints, 3*Natoms, 3*Natoms].
        self.displacements: Displacement vectors. np.array.
            [Nqpoints, 3*Natoms, 3*Natoms].
        self.frequencies: Square root of the eigenvalues of the dynamical
            matrix in units of cm-1. np.array. [Nqpoints, 3*Natoms].

    """

    def __init__(self, file):
        data = (open(file, "r")).readlines()
        self.Nspecies = int((data[2].split())[0])
        self.Natoms = int((data[2].split())[1])
        self.alat = float((data[2].split())[3]) * consts.bohr2angstroms
        tmp_atom_coords = np.empty([self.Natoms, 3])
        tmp_cell = np.empty([3, 3])
        self.masses = np.empty([self.Natoms])
        atomic_species_dict = {}
        atomic_masses_dict = {}
        tmp_atom_species = np.empty([self.Natoms, 3], dtype="<U5")
        for i in range(3):
            for j in range(3):
                tmp_cell[i, j] = float((data[4 + i].split())[j])
        tmp_cell = self.alat * tmp_cell
        self.rcell = cu.get_rcell(tmp_cell, alat=self.alat)  # alat units
        for i in range(self.Nspecies):
            atomic_species_dict[data[7 + i].split()[1][1:]] = data[7 + i].split()[0]
            atomic_masses_dict[data[7 + i].split()[0]] = data[7 + i].split()[3]
        for atom in range(self.Natoms):
            tmp_atom_coords[atom, 0] = float(data[7 + self.Nspecies + atom].split()[2])
            tmp_atom_coords[atom, 1] = float(data[7 + self.Nspecies + atom].split()[3])
            tmp_atom_coords[atom, 2] = float(data[7 + self.Nspecies + atom].split()[4])
            tmp_atom_species[atom, 1] = data[7 + self.Nspecies + atom].split()[1]
            key_list = list(atomic_species_dict.keys())
            val_list = list(atomic_species_dict.values())
            pos = val_list.index(tmp_atom_species[atom, 1])
            tmp_atom_species[atom, 0] = key_list[pos]
            tmp_atom_species[atom, 2] = ase.data.atomic_numbers[
                str(tmp_atom_species[atom][0])
            ]
            self.masses[atom] = atomic_masses_dict[tmp_atom_species[atom, 1]]
        tmp_atom_coords = tmp_atom_coords * self.alat
        self.structure = get_struct(
            tmp_cell, tmp_atom_coords, tmp_atom_species, self.Nspecies
        )
        data1 = []
        for i in range(len(data)):
            try:
                if data[i].split()[0] != "\n":
                    data1.append(data[i])
            except:
                None
        data = data1
        self.Nqpoints = -1
        for i in range(len(data)):
            if data[i].split()[0] == "q":
                self.Nqpoints += 1
        self.qpoints = np.empty([self.Nqpoints, 3])
        i = 3
        self.dyn_matrixes = np.empty(
            [self.Nqpoints, self.Natoms, self.Natoms, 3, 3], dtype=complex
        )
        for q_index in range(self.Nqpoints):
            while (data[i][:]).split()[0] != "q":
                i += 1
            self.qpoints[q_index] = np.array([(data[i][:]).split()[3:6]])
            i += 2
            for n1 in range(self.Natoms):
                for n2 in range(self.Natoms):
                    for n3 in range(3):
                        self.dyn_matrixes[q_index, n1, n2, n3, 0] = complex(
                            float((data[i + n3].split())[0]),
                            float((data[i].split())[1]),
                        )
                        self.dyn_matrixes[q_index, n1, n2, n3, 1] = complex(
                            float((data[i + n3].split())[2]),
                            float((data[i].split())[3]),
                        )
                        self.dyn_matrixes[q_index, n1, n2, n3, 2] = complex(
                            float((data[i + n3].split())[4]),
                            float((data[i].split())[5]),
                        )
                    i += 4
        self.polarization_vectors = np.empty(
            [self.Nqpoints, self.Natoms * 3, self.Natoms * 3], complex
        )
        self.frequencies = np.empty([self.Nqpoints, self.Natoms * 3], complex)
        self.displacements = np.empty(
            [self.Nqpoints, self.Natoms * 3, self.Natoms * 3], complex
        )
        self.real_dyns = QEtoREAL_dyn(self)
        for q in range(self.Nqpoints):
            v, e = diago_dyn_matrixes(self.real_dyns[q])
            (
                self.frequencies[q, :],
                self.polarization_vectors[q, :],
                self.displacements[q, :],
            ) = reorder(self, v, e)
        self.frequencies = np.sqrt(self.frequencies) * consts.Ry2cm

    def expand_polvecs_displ(self, qf_list=None, mod=None, q=-1):
        """This function generates the expanded polarization vectors
        and displacements.

        Input:
            qf_list: Contains the quantum numbers q and lambda(f). q indicates the point
                in reciprocal space (the index that is read from dyn file), and f the
                frequency starting from 0. list of list.
            mod: Modulation of the outcell. np.array or list.
            q: The q-point id for which to expand all the polarization vectors. If not
                specified just the polarization vectors of qf_list are expanded.

        Returns:
            e_list: Expanded polarization vectors. list of np.array. [Number of modes, Natoms*3*mod[0]*mod[1]*mod[2]].
            disp_list: Expanded displacement vectors. list of np.array. [Number of modes, Natoms*3*mod[0]*mod[1]*mod[2]].
            masses: Expanded masses. np.array. [Natoms*3*mod[0]*mod[1]*mod[2]]
        """
        if q == -1:
            masses = np.empty([self.Natoms * mod[0] * mod[1] * mod[2] * 3], complex)
            e_list = []
            disp_list = []
            for qf in qf_list:
                expanded_pol_vecs = np.empty(
                    [self.Natoms * mod[0] * mod[1] * mod[2] * 3], complex
                )
                expanded_displ = np.empty(
                    [self.Natoms * mod[0] * mod[1] * mod[2] * 3], complex
                )
                for i in range(mod[0]):
                    for j in range(mod[1]):
                        for k in range(mod[2]):
                            expanded_pol_vecs[
                                self.Natoms
                                * 3
                                * (mod[1] * mod[2] * i + mod[2] * j + k) : self.Natoms
                                * 3
                                * (mod[1] * mod[2] * i + mod[2] * j + k + 1)
                            ] = self.polarization_vectors[qf[0], qf[1]] * np.exp(
                                2j
                                * np.pi
                                * np.matmul(
                                    cu.cart2cryst(self.qpoints[qf[0]], self.rcell),
                                    np.array([i, j, k]),
                                )
                            )
                            for atom in range(self.Natoms):
                                masses[
                                    self.Natoms
                                    * 3
                                    * (mod[1] * mod[2] * i + mod[2] * j + k)
                                    + atom
                                    * 3 : self.Natoms
                                    * 3
                                    * (mod[1] * mod[2] * i + mod[2] * j + k)
                                    + (atom + 1) * 3
                                ] = self.masses[atom]
                expanded_pol_vecs = expanded_pol_vecs / np.linalg.norm(
                    expanded_pol_vecs
                )
                ##############################################################################
                ##### The displacements are constructed from already normalized pol vecs #####
                ##############################################################################
                for i in range(self.Natoms * mod[0] * mod[1] * mod[2] * 3):
                    expanded_displ[i] = expanded_pol_vecs[i] / np.sqrt(masses[i])
                e_list.append(expanded_pol_vecs)
                disp_list.append(expanded_displ)
        else:
            qf_list = np.empty([self.Natoms * 3, 2], dtype=int)
            for a in range(3 * self.Natoms):
                qf_list[a, 0] = q
                qf_list[a, 1] = a
            masses = np.empty([self.Natoms * mod[0] * mod[1] * mod[2] * 3], complex)
            e_list = []
            disp_list = []
            for qf in qf_list:
                expanded_pol_vecs = np.empty(
                    [self.Natoms * mod[0] * mod[1] * mod[2] * 3], complex
                )
                expanded_displ = np.empty(
                    [self.Natoms * mod[0] * mod[1] * mod[2] * 3], complex
                )
                for i in range(mod[0]):
                    for j in range(mod[1]):
                        for k in range(mod[2]):
                            expanded_pol_vecs[
                                self.Natoms
                                * 3
                                * (mod[1] * mod[2] * i + mod[2] * j + k) : self.Natoms
                                * 3
                                * (mod[1] * mod[2] * i + mod[2] * j + k + 1)
                            ] = self.polarization_vectors[qf[0], qf[1]] * np.exp(
                                2j
                                * np.pi
                                * np.matmul(
                                    cu.cart2cryst(self.qpoints[qf[0]], self.rcell),
                                    np.array([i, j, k]),
                                )
                            )
                            for atom in range(self.Natoms):
                                masses[
                                    self.Natoms
                                    * 3
                                    * (mod[1] * mod[2] * i + mod[2] * j + k)
                                    + atom
                                    * 3 : self.Natoms
                                    * 3
                                    * (mod[1] * mod[2] * i + mod[2] * j + k)
                                    + (atom + 1) * 3
                                ] = self.masses[atom]
                expanded_pol_vecs = expanded_pol_vecs / np.linalg.norm(
                    expanded_pol_vecs
                )
                ##############################################################################
                ##### The displacements are constructed from already normalized pol vecs #####
                ##############################################################################
                for i in range(self.Natoms * mod[0] * mod[1] * mod[2] * 3):
                    expanded_displ[i] = expanded_pol_vecs[i] / np.sqrt(masses[i])
                e_list.append(expanded_pol_vecs)
                disp_list.append(expanded_displ)
        return e_list, disp_list, masses

    def distort_structure(self, Q, qf_list, mod):
        """This function distorts an structure from a dyn_matrix.
        OP must coincide with the dimension of the qpoints of the
        dyn_matrix.

        Input:
            Q: Order-parameter. np.array or list. Same dimensionality as rows of qf_list.
            qf_list: Contains the quantum numbers q and lambda(f). q indicates the point
                in reciprocal space (the index that is read from dyn file), and f the
                frequency starting from 0. list of list.
            mod: Modulation of the outcell. np.array or list.

        Returns:
            struct: STRUCT object.
        """
        struct = STRUCT()
        tmp_cell = np.empty([3, 3])
        for i in range(3):
            tmp_cell[i, :] = self.structure.cell[i, :] * mod[i]
        tmp_atom_coords = np.empty([self.Natoms * mod[0] * mod[1] * mod[2], 3])
        tmp_atom_species = np.empty(
            [self.Natoms * mod[0] * mod[1] * mod[2], 3], dtype="<U5"
        )
        for i in range(mod[0]):
            for j in range(mod[1]):
                for k in range(mod[2]):
                    tmp_atom_coords[
                        self.Natoms
                        * (4 * i + 2 * j + k) : self.Natoms
                        * (4 * i + 2 * j + k + 1)
                    ] = self.structure.atom_coords + (
                        self.structure.cell[0, :] * i
                        + self.structure.cell[1, :] * j
                        + self.structure.cell[2, :] * k
                    )
                    tmp_atom_species[
                        self.Natoms
                        * (4 * i + 2 * j + k) : self.Natoms
                        * (4 * i + 2 * j + k + 1)
                    ] = self.structure.atom_species
        tmp_disp_list = (self.expand_polvecs_displ(qf_list=qf_list, mod=mod))[1]
        for i, mode in enumerate(tmp_disp_list):
            rs_tmp_displ = np.reshape(np.real(mode), tmp_atom_coords.shape)
            tmp_atom_coords += Q[i] * rs_tmp_displ
        struct.set_atom_coords(tmp_atom_coords, units="angstrom")
        struct.set_cell(tmp_cell, units="angstrom")
        struct.set_atom_species(tmp_atom_species)
        struct.set_Nspecies(self.Nspecies)
        return struct


def get_struct(cell, atom_coords, atom_species, Nspecies):
    """With this function, all the methods of the STRUCT
    class are also available for the dyn matrix.

    Input:
        cell: Unit cell. np.array. [3, 3].
        atom_coords: Atomic-coordinates. np.array. [Natoms, 3].
        atom_species: Atomic species. np.array. [Natoms, 3].
        Nspecies: Number of different species. int.

    Returns:
        struct: STRUCT object.

    """
    struct = STRUCT()
    struct.set_cell(cell, units="angstrom")
    struct.set_atom_coords(atom_coords, units="angstrom")
    struct.set_atom_species(atom_species)
    struct.set_Nspecies(Nspecies)
    return struct


def diago_dyn_matrixes(dyn_matrix):
    """Diagonalize the dynamical matrix.

    Returns:
        e: Polarization vectors.
        v: Eigenvalues.
    """
    D = construct_matrix(dyn_matrix)
    e, v = np.linalg.eig(D)
    return (e, v)


def construct_matrix(dyn_matrix):
    """Reformat the dynamical matrix from [Natoms, Natoms, 3, 3]
    into, [Natoms*3, Natoms*3].

    Returns:
        D: Dynamical matrix.
    """
    dim = len(dyn_matrix[:, 0, 0, 0]) * 3
    D = np.empty([dim, dim], dtype=complex)
    for i in range(dim):
        for j in range(dim):
            n1 = i // 3
            n2 = j // 3
            n3 = i % 3
            n4 = j % 3
            D[i, j] = dyn_matrix[n1, n2, n3, n4]
    return D


def QEtoREAL_dyn(DYN_MATRIX):
    """Return the dynamical matrix from the force constants printed
    by Quantum Espresso.

    Input:
        DYN_MATRIX: The object DYN_MATRIX.

    Returns:
        real_dyns = Dynamical matrixes of each q-point.
    """
    real_dyns = np.empty(
        [DYN_MATRIX.Nqpoints, DYN_MATRIX.Natoms, DYN_MATRIX.Natoms, 3, 3], dtype=complex
    )
    for q_index in range(DYN_MATRIX.Nqpoints):
        for n1 in range(DYN_MATRIX.Natoms):
            for n2 in range(DYN_MATRIX.Natoms):
                real_dyns[q_index, n1, n2, :, :] = DYN_MATRIX.dyn_matrixes[
                    q_index, n1, n2, :, :
                ] / np.sqrt(DYN_MATRIX.masses[n1] * DYN_MATRIX.masses[n2])
    return real_dyns


def POLtoDISP(DYN_MATRIX, e):
    """From polarization into displacement vectors.

    Input:
        DYN_MATRIX: DYN_MATRIX object.
        e: Polarization vector. np.array. [3*Natoms, 3*Natoms].

    Returns:
        disp: Displacement vector. np.array. [3*Natoms, 3*Natoms].
    """
    disp = np.empty(e.shape)
    for mode in range(DYN_MATRIX.Natoms * 3):
        for atom in range(DYN_MATRIX.Natoms):
            i = 3 * atom
            disp[i : i + 3, mode] = np.real(
                e[i : i + 3, mode] / np.sqrt(DYN_MATRIX.masses[atom])
            )
    return disp


def reorder(DYN_MATRIX, v, e):
    """Reorders the eingenvalues and the eigensolutions from
    lowest to highest frequencies.

    Input:
        DYN_MATRIX: DYN_MATRIX object.
        v: Frequencies. np.array. [3*Natoms].
        e: Polarization vector. np.array. [3*Natoms, 3*Natoms].

    Returns:
        freqs: Ordered frequencies. np.array. [3*Natoms].
        pol: Ordered polarization vectors. np.array. [3*Natoms, 3*Natoms].
        disp: Ordered displacement vectors. np.array. [3*Natoms, 3*Natoms].
    """
    e = e.transpose()
    v = v.transpose()
    freqs = np.empty(len(v), complex)
    tmp_disp = POLtoDISP(DYN_MATRIX, e)
    disp = np.empty([len(e[:, 0]), len(e[:, 0])], complex)
    pol = np.empty([len(e[:, 0]), len(e[:, 0])], complex)
    order = np.argsort(v)
    for i, n in enumerate(order):
        freqs[i] = v[n]
        disp[i] = tmp_disp[n]
        pol[i] = e[n]
    return freqs, pol, disp

##################################################################
##                         DEVELOPMENT                          ##
##################################################################


def project_into_polvecs(du, expanded_pol_vecs, expanded_masses):
    """Theoretically du is an imaginary number.
    du: np.array. length of array = Natoms*3
    expanded_pol_vecs: list of different pol vectors in np.array. length of array = Natoms*3.
    expanded_masses: np.array. length of array = Natoms*3
    """
    du = np.array(du, dtype=complex)
    Q = np.zeros(len(expanded_pol_vecs), dtype=complex)
    for i, vec in enumerate(expanded_pol_vecs):
        for j in range(len(expanded_masses)):
            Q[i] += np.sqrt(expanded_masses[j]) * np.conjugate(vec[j]) * du[j]
    return Q


def read_dyns(file, nqirr=0):
    """This function reads a set of dynamical matrixes (or just
    GAMMA by default), and returns a list of them."""
    dyns = []
    if nqirr == 0:
        dyns.append(DYN_MATRIX(f"{file}"))
    else:
        for i in range(nqirr):
            dyns.append(DYN_MATRIX(f"{file}{i+1}"))
    return dyns


def write_in_Q_basis(dyn_matrixes, pol_vecs, masses, all=True, i1=0, j1=0):
    """The order parameter basis corresponds to the polarization vector."""
    Natoms = len(dyn_matrixes[:, 0, 0, 0])
    tmp_dyn = np.zeros([len(pol_vecs), len(pol_vecs)], dtype=complex)
    mat = np.empty([len(pol_vecs), len(pol_vecs)], dtype=complex)
    for i in range(len(mat[:, 0])):
        for j in range(len(mat[0, :])):
            # To define the matrix for basis transformation, the
            # displacement vector must correspond to columns.
            mat[j, i] = pol_vecs[i, j]
    mat_inv = np.linalg.inv(mat)
    if all:
        for i in range(len(pol_vecs)):
            for j in range(len(pol_vecs)):
                print(f"{i}, {j}")
                for a in range(Natoms):
                    for b in range(Natoms):
                        for alpha in range(3):
                            for beta in range(3):
                                norm = np.linalg.norm(pol_vecs[i, :]) * np.linalg.norm(
                                    pol_vecs[j, :]
                                )
                                Aai, Abj = (
                                    mat_inv[i, 3 * a + alpha],
                                    mat_inv[j, 3 * b + beta],
                                )
                                tmp_dyn[i, j] += (
                                    1
                                    / np.sqrt(masses[3 * a] * masses[3 * b])
                                    * dyn_matrixes[a, b, alpha, beta]
                                    * Aai
                                    * Abj
                                    / norm
                                )
    else:
        for a in range(Natoms):
            for b in range(Natoms):
                for alpha in range(3):
                    for beta in range(3):
                        norm = np.linalg.norm(pol_vecs[i1, :]) * np.linalg.norm(
                            pol_vecs[j1, :]
                        )
                        Aai, Abj = mat_inv[i1, 3 * a + alpha], mat_inv[j1, 3 * b + beta]
                        tmp_dyn[i1, j1] += (
                            1
                            / np.sqrt(masses[3 * a] * masses[3 * b])
                            * dyn_matrixes[a, b, alpha, beta]
                            * Aai
                            * Abj
                            * norm
                        )
    return tmp_dyn


def test_write_in_Q_basis(dyn_matrixes, displacements, masses, all=True, i1=0, j1=0):
    """Test to see if write_in_Q_basis works. We know already the eigenvalues of
    the polarization vector from the dynamical matrix. If the Q basis corresponds
    to the polarization vectors, the FCs must be diagonal and have the same value
    as the original FCs. dyn.frequencies_1 is given in Ry also to compare."""
    Natoms = len(dyn_matrixes[:, 0, 0, 0])
    tmp_dyn = np.zeros([len(displacements), len(displacements)], dtype=complex)
    mat = np.empty([len(displacements), len(displacements)], dtype=complex)
    for i in range(len(mat[:, 0])):
        for j in range(len(mat[0, :])):
            # To define the matrix for basis transformation, the
            # displacement vector must correspond to columns.
            mat[j, i] = displacements[i, j]
    mat_inv = np.linalg.inv(mat)
    if all:
        for i in range(len(displacements)):
            for j in range(len(displacements)):
                print(f"{i}, {j}")
                for a in range(Natoms):
                    for b in range(Natoms):
                        for alpha in range(3):
                            for beta in range(3):
                                norm = np.linalg.norm(
                                    displacements[i, :]
                                ) * np.linalg.norm(displacements[j, :])
                                Aai, Abj = (
                                    mat_inv[i, 3 * a + alpha],
                                    mat_inv[j, 3 * b + beta],
                                )
                                tmp_dyn[i, j] += (
                                    1
                                    / np.sqrt(masses[3 * a] * masses[3 * b])
                                    * dyn_matrixes[a, b, alpha, beta]
                                    * Aai
                                    * Abj
                                    / norm
                                )
    else:
        for a in range(Natoms):
            for b in range(Natoms):
                for alpha in range(3):
                    for beta in range(3):
                        norm = np.linalg.norm(displacements[i1, :]) * np.linalg.norm(
                            displacements[j1, :]
                        )
                        Aai, Abj = mat_inv[i1, 3 * a + alpha], mat_inv[j1, 3 * b + beta]
                        tmp_dyn[i1, j1] += (
                            1
                            / np.sqrt(masses[3 * a] * masses[3 * b])
                            * dyn_matrixes[a, b, alpha, beta]
                            * Aai
                            * Abj
                            * norm
                        )
    return tmp_dyn
