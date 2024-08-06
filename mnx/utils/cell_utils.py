import numpy as np


def cryst2cart(xyz, cell, alat=False):
    """Transforms from crystalline to cartesian units.

    Input:
        xyz: The coordinates in fractional units. np.array. [X, 3].
        cell: The unit cell. np.array. [3, 3].
        alat: If TRUE the cartesian coordinates are given in alat units. boolean.

    Returns:
        coords_crystal: Atomic coordinates in cartesian units. np.array. [X, 3].
    """
    if alat:
        M = cell / np.linalg.norm(cell[0, :])
        coords_crystal = np.matmul(M.transpose(), xyz.transpose()).transpose()
    else:
        M = cell
        coords_crystal = np.matmul(M.transpose(), xyz.transpose()).transpose()
    return coords_crystal


def cart2cryst(xyz, cell):
    """Transforms from crystalline to cartesian units.

    Input:
        xyz: The coordinates in cartesian units. np.array. [X, 3].
        cell: The unit cell. np.array. [3, 3].

    Returns:
        coords_crystal: Atomic coordinates in crystalline units. np.array. [X, 3].
    """
    M = np.linalg.inv(cell.transpose())
    coords_crystal = np.matmul(M, xyz.transpose()).transpose()
    return coords_crystal


def get_rcell(cell, alat=1):
    """Calculate the reciprocal cell.

    Input:
        cell: The unit cell. np.array. [3, 3].

    Returns:
        rcell: Reciprocal cell. np.array. [3, 3].
    """
    rcell = np.empty([3, 3])
    for k in range(3):
        s = np.zeros([3])
        s[k] = 1
        rcell[k, :] = np.matmul(np.linalg.inv(cell / alat), s)
    return rcell


def corrected_displacements(struct, ref_struct):
    """Gives displacements of atoms in the case that atoms jump from one
    cell to another.

    Input:
        struct: STRUCT object.
        ref_struct: Reference STRUCT object respect to whom calculate the displacements.

    Returns:
        du: Corrected displacements. np.array. [Natoms, 3].
    """
    du = struct.atom_coords - ref_struct.atom_coords
    du = np.reshape(cart2cryst(du, struct.cell), struct.Natoms * 3)
    for i, dr in enumerate(du):
        if dr > 0.5:
            du[i] = du[i] - 1
        elif dr < -0.5:
            du[i] = du[i] + 1
    du = np.reshape(du, [struct.Natoms, 3])
    return du


def map2structure(struct, ref_struct, R=0.2):
    """This function maps two atomic structures and returns the id list (with the
    info of the mapping). This id_list can be then used to reorder structures or vectors.

    Input:
        ref_struct: Reference STRUCT object.
        R: Max displacement in angstroms, between two atoms to be equivalent.

    Returns:
        None
    """
    id_list = np.empty([struct.Natoms], dtype=int)
    translations = np.zeros([struct.Natoms, 3])
    for i, pos in enumerate(struct.atom_coords):
        found = False
        for j, ref_pos in enumerate(ref_struct.atom_coords):
            dr = np.sum((pos - ref_pos) ** 2)
            if dr < R:
                id_list[i] = j
                found = True
            else:
                for T1 in [0, -1, 1]:
                    for T2 in [0, -1, 1]:
                        for T3 in [0, -1, 1]:
                            if not found:
                                dr = np.sum(
                                    np.square(
                                        pos
                                        - ref_pos
                                        + T1 * struct.cell[0]
                                        + T2 * struct.cell[1]
                                        + T3 * struct.cell[2]
                                    )
                                )
                                if dr < R:
                                    id_list[i] = j
                                    translations[i] = [T1, T2, T3]
                                    found = True
                                    break
                if not found:
                    for T1 in [-2, -1, 0, 1, 2]:
                        for T2 in [-2, -1, 0, 1, 2]:
                            for T3 in [-2, -1, 0, 1, 2]:
                                if not found:
                                    dr = np.sum(
                                        np.square(
                                            pos
                                            - ref_pos
                                            + T1 * struct.cell[0]
                                            + T2 * struct.cell[1]
                                            + T3 * struct.cell[2]
                                        )
                                    )
                                    if dr < R:
                                        id_list[i] = j
                                        translations[i] = [T1, T2, T3]
                                        found = True
                                        break
                if not found:
                    for T1 in [-3, -2, -1, 0, 1, 2, 3]:
                        for T2 in [-3, -2, -1, 0, 1, 2, 3]:
                            for T3 in [-3, -2, -1, 0, 1, 2, 3]:
                                if not found:
                                    dr = np.sum(
                                        np.square(
                                            pos
                                            - ref_pos
                                            + T1 * struct.cell[0]
                                            + T2 * struct.cell[1]
                                            + T3 * struct.cell[2]
                                        )
                                    )
                                    if dr < R:
                                        id_list[i] = j
                                        translations[i] = [T1, T2, T3]
                                        found = True
                                        break
        if not found:
            print(f"Translation of atom {i} not found up to 3 cells.")
    return (id_list, translations)


def reorder2list(struct, id_list, translations):
    """Reorder the atomic id from a list. The list and translations
    are obtained with the function map2structure in the utils/cell_utils.py.

    Input:
        struct:
        id_list: List of atomic id. list. [Natoms].
        translations: Translations corresponding to the atom id. [Natoms, 3].

    Returns:
        None
    """
    tmp_atom_coords = np.empty([struct.Natoms, 3])
    tmp_atom_forces = np.empty([struct.Natoms, 3])
    tmp_atom_species = np.empty([struct.Natoms, 3], dtype="<U5")
    for i in range(struct.Natoms):
        tmp_atom_coords[i] = struct.atom_coords[id_list[i]] - cryst2cart(
            translations[i], struct.cell
        )
        tmp_atom_forces[i] = struct.atom_forces[id_list[i]]
        tmp_atom_species[i] = struct.atom_species[id_list[i]]
    struct.set_atom_coords(tmp_atom_coords)
    struct.set_atom_species(tmp_atom_species)
    struct.set_atom_forces(tmp_atom_forces)
    return struct


# def reorder2list_vec(vec, id_list):
#     tmp_vec = np.empty(len(vec), dtype=complex)
#     for a in range(int(len(vec) / 3)):
#         for alpha in range(3):
#             tmp_vec[3 * a + alpha] = vec[3 * id_list[a] + alpha]
#     return tmp_vec


# def reorder2list_FC(FC, id_list):
#     """"""
#     Natoms = len(FC[:, 0, 0, 0])
#     tmp_dyn = np.empty([Natoms, Natoms, 3, 3], dtype=complex)
#     for a in range(Natoms):
#         for b in range(Natoms):
#             tmp_dyn[a, b, :, :] = FC[id_list[a], id_list[b], :, :]
#     return tmp_dyn


# def reshape_FC(FC):
#     Natoms = int(len(FC[:, 0]) / 3)
#     tmp_FC = np.empty([Natoms, Natoms, 3, 3], dtype=complex)
#     for a in range(Natoms):
#         for b in range(Natoms):
#             for alpha in range(3):
#                 for beta in range(3):
#                     tmp_FC[a, b, alpha, beta] = FC[3 * a + alpha, 3 * b + beta]
#     return tmp_FC
