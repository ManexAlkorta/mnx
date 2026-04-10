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

def get_reccellpos(rot_cell, ref_cell, symprec):
    rel_cell_pos = np.zeros([3, 3])
    for i, pos in enumerate(rot_cell):
        tmp_pos = cryst2cart(cart2cryst(pos, ref_cell)%1,ref_cell)
        found = False
        for j, ref_pos in enumerate(ref_cell):
            tmp_ref_pos = cryst2cart(cart2cryst(ref_pos, ref_cell)%1,ref_cell)
            dr = np.sqrt(np.sum((tmp_pos - tmp_ref_pos) ** 2))
            if dr < symprec:
                rel_cell_pos[i] = tmp_pos-tmp_ref_pos
                found = True
                break
        if not found:
            tmp_pos = cryst2cart(cart2cryst(pos+1, ref_cell)%1,ref_cell)
            for j, ref_pos in enumerate(ref_cell):
                tmp_ref_pos = cryst2cart(cart2cryst(ref_pos+1, ref_cell)%1,ref_cell)
                dr = np.sqrt(np.sum((tmp_pos - tmp_ref_pos) ** 2))
                if dr < symprec:
                    rel_cell_pos[i] = tmp_pos-tmp_ref_pos
                    found = True
                    break
            if not found:
                breakpoint()
                print(f"Cell {i} not found")
    return rel_cell_pos

def map2structure(struct0, struct1, symprec):
    """This function maps two atomic structures and returns the id list (with the
    info of the mapping). This id_list can be then used to reorder structures or vectors.

    Input:
        ref_struct: Reference STRUCT object.
        R: Max displacement in angstroms, between two atoms to be equivalent.

    Returns:
        None
    """
    struct, ref_struct = struct1, struct0 # This should be fixed!!
    id_list = np.empty([struct.Natoms], dtype=int)
    rel_pos = np.zeros([struct.Natoms, 3])
    for i, pos in enumerate(struct.atom_coords):
        found = False
        for j, ref_pos in enumerate(ref_struct.atom_coords):
            dr = np.sqrt(np.sum((pos - ref_pos) ** 2))
            if dr < symprec:
                id_list[i] = j
                rel_pos[i] = pos-ref_pos
                found = True
                break
        if not found:
            tmp_struct = struct.copy()
            tmp_struct.atom_coords += symprec
            tmp_struct._fix_coords()
            pos = tmp_struct.atom_coords[i]
            tmp_ref_struct = ref_struct.copy()
            tmp_ref_struct.atom_coords += symprec
            tmp_ref_struct._fix_coords()
            pos = tmp_struct.atom_coords[i]
            for j, ref_pos in enumerate(tmp_ref_struct.atom_coords):
                dr = np.sqrt(np.sum((pos - ref_pos) ** 2))
                if dr < symprec:
                    id_list[i] = j
                    rel_pos[i] = pos-ref_pos
                    found = True
                    break
            if not found:
                print(f"Atom {i} not found")
    return (id_list, rel_pos)


def reorder2list(struct, id_list, rel_pos):
    """Reorder the atomic id from a list. The list and translations
    are obtained with the function map2structure in the utils/cell_utils.py.

    Input:
        struct:
        id_list: List of atomic id. list. [Natoms].

    Returns:
        None
    """
    tmp_atom_coords = np.empty([struct.Natoms, 3])
    tmp_atom_species = np.empty([struct.Natoms, 2], dtype="<U5")
    for i in range(struct.Natoms):
        tmp_atom_coords[i] = struct.atom_coords[id_list[i]] - rel_pos[i]
        tmp_atom_species[i] = struct.atom_species[id_list[i]]
    struct.atom_coords = tmp_atom_coords
    struct._set_atom_species(tmp_atom_species)
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
