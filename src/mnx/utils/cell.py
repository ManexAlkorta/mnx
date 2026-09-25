import numpy as np
import spglib

import cellconstructor 

def cryst2cart(xyz, cell, alat=False):
    """Transforms from crystalline to cartesian units.

    Input:
        xyz: The coordinates in fractional units. np.array. [X, 3].
        cell: The unit cell. np.array. [3, 3].
        alat: If TRUE the cartesian coordinates are given in alat units. boolean.

    Returns:
        coords_crystal: Atomic coordinates in cartesian units. np.array. [X, 3].
    """
    xyz = np.asarray(xyz)
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
    xyz = np.asarray(xyz)
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

################################################################

def matrix_cart2cryst(matrix, cell):
    """
    """
    
    
    # Get the metric tensor from the unit_cell
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = cell[i,:].dot(cell[j, :])

    # Choose which conversion perform
    comp_matrix = np.einsum("ij, jk", np.linalg.inv(metric_tensor), cell) 
    comp_matrix_inv = np.linalg.inv(comp_matrix)

    return comp_matrix.dot( np.dot(matrix, comp_matrix_inv))

def matrix_cryst2cart(matrix, cell):
    """
    """
    
    
    # Get the metric tensor from the unit_cell
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = cell[i,:].dot(cell[j, :])

    # Choose which conversion perform
    comp_matrix = np.einsum("ij, jk", np.linalg.inv(metric_tensor), cell) 
    comp_matrix_inv = np.linalg.inv(comp_matrix)

    return comp_matrix_inv.dot( np.dot(matrix, comp_matrix))

def get_lg_idx(structure, q_point, symprec = 1e-5):
    # Write q_point in our convention.
    q_point_cart = cryst2cart(q_point, structure.rcell)
    q_point_cart = map_q_to_1st_bz(structure.rcell, q_point_cart)
    q_point = cart2cryst(q_point_cart, structure.rcell)

    syms = spglib.get_symmetry(structure.get_spglib_cell(), symprec=symprec)
    R, T = syms["rotations"], syms["translations"]

    lg_idx = []
    for ri in range(len(R)):
        rot_cart_q_point = cryst2cart(R[ri] @ q_point, structure.rcell)
        rot_cart_q_point = map_q_to_1st_bz(structure.rcell, rot_cart_q_point)
        if np.all(np.abs(rot_cart_q_point - q_point_cart)<1e-6):
            lg_idx.append(ri)

    return lg_idx

def map_qsinglet(q_list, q_list_frac, rcell, rot_cart):
    """
    This functions performs the mapping between the commensurate wave-vectors of the
    supercell. 
    
    Parameters
    ----------
        - q_list: np.ndarray
            List of wave-vectors in cartesian coordinates. Dimension [Nq,3].
        - q_list_frac: np.ndarray
            List of wave-vectors in fractional coordinates. Dimension [Nq,3].
        - rcell: np.ndarray
            Reciprocal unit-cell. Dimension [3,3].
        - rot_cart: np.ndarray
            Point symmetry in cartesian coordinates. Dimension [Nsym,3,3].
    
    Returns
    -------
        - mapping: np.ndarray
            The mapping between symmetry related wave-vectors for each point
            point symmetry of the crystal. Dimension [Nq,Nsym]
        - orbit1a: np.ndarray
            Wave-vector classification in orbits. The classification is not
            unique in this case, because needs to be used to construct the P
            matrix (symmetry related elements appear more than once).
            Dimension [Nq,tbd,1].
        - orbit1s: np.ndarray
            Which symmetry makes the classification in orbits/stars of orbit1a.
            Dimension [Nq,tbd,1].
        - norbit: np.ndarray
            Number of wave-vectors in orbit. Dimension [Nrefq1].
        - its_zb: np.ndarray
            Checks whether the points in q_list are on zone border or not. This
            is later used to impose time-reversal symmetry.
    """

    q_list_frac_fixed = np.empty(q_list_frac.shape)
    its_zb = np.empty(q_list_frac.shape[0], dtype=np.int8)
    for qi,q in enumerate(q_list):
        q_list_frac_fixed[qi] = np.round(cart2cryst(map_q_to_1st_bz(rcell, q), rcell),6)
        if np.all((np.abs(np.round(2*q_list_frac_fixed[qi], 6))%1)<1e-6):
            its_zb[qi] = 0
        else:
            for alpha in range(3):
                if np.abs(q_list[qi,alpha])>1e-6:
                    if q_list[qi,alpha] > 1e-6:
                        its_zb[qi] = 1 # Non zone border. Positive class.
                        break
                    else:
                        its_zb[qi] = 2 # Non zone border. Negative class.
                        break

    mapping = np.zeros([len(q_list), rot_cart.shape[0]], dtype=np.int32)
    for qi, q in enumerate(q_list):
        for isym in range(rot_cart.shape[0]):
            q_sym_cart = rot_cart[isym] @ q
            q_sym_cart_1bz = map_q_to_1st_bz(rcell, q_sym_cart)
            q_sym = np.round(cart2cryst(q_sym_cart_1bz, rcell),6)
            match = np.all(np.abs(q_list_frac_fixed-q_sym)<1e-3, axis=1)
            qii = np.where(match)
            mapping[qi,isym] = qii[0][0]

    orbit1a = np.zeros([len(q_list),rot_cart.shape[0],1], dtype=np.int32)
    orbit1s = np.zeros([len(q_list),rot_cart.shape[0],1], dtype=np.int32)

    # Loop over all q-points, and knowing the mapping build the star for each q-point
    # All the classifications are considered (equivalents are not excluded in next iters)

    for qi in range(len(q_list)):
        for isym,qsym in enumerate(mapping[qi,:]):
            orbit1a[qi,isym,0] = qsym
            orbit1s[qi,isym,0] = isym
    return mapping, orbit1a, orbit1s, its_zb

def map_singlet(dyn, symprec=1e-5, verbose=False):
    """
    Classifies atomic singlets, and returns the atomic map for each symmetry operation.

    Parameters
    ----------
        - dyn: object
            Cellconstructor dynamical matrix.
        - symprec: float
            Tolerance parameter for spglib in symmetry detection.
            Defaults to 1e-5.
        - verbose: bool
            If True prints information during execution.
            Defaults to False.
    
    Returns
    -------
        - mapping: np.ndarray
            Atomic mapping for each symmetry operation. Dimension [Natom_sc,Nsym].
        - orbit1s: np.ndarray
            Symmetry mapping for each atomic mapping. Dimension [Natom_sc,Nsym].
        - rot_cart: np.ndarray
            Variable containing all the symmetry operations of the crystal. Dimension [Nsym,3,3]
        - map_uc: np.ndarray
            An array that works as a tool to know which index i we need to consider, taking
            index j as reference (unit-cell). Dimension [Natom_sc, Natom_sc].
        - map_tr: np.ndarray
            This array says which is the translation employed to map atom index i using index
            j as reference (unit-cell). Dimension [Natom_sc, Natom_sc].
        - T_list: np.ndarray
            Contains all the translations in unit-cell fractional units. Dimension [Nq,3].
        - T_list_frac: np.ndarray
            Contains all the translations in supercell fractional units. Dimension [Nq,3].
    """

    sg = spglib.get_spacegroup(dyn.structure.get_spglib_cell(), symprec)

    if verbose:
        print("Initial SG=", sg)
        print("===== STARTING SINGLET CLASSIFICATION =====")

    spg_syms = spglib.get_symmetry(dyn.structure.get_spglib_cell(), symprec)
    sym_uc = cellconstructor.symmetries.GetSymmetriesFromSPGLIB(spg_syms, regolarize=False)

    Nsym = len(sym_uc)

    # Obtain number of supercells. GetSupercell returns the modulation as 
    # 3 dimensional list, sc_size. 
    sc_size = dyn.GetSupercell() 
    Nsupercell = np.prod(sc_size)

    # Obtain point group symmetries in cryst coord respect sc: 
    sym_list = np.zeros((Nsym,3,4),dtype=np.float64)
    for isym in range(Nsym):
        sym_list[isym,:,:] = sym_uc[isym]
        for ll in range(3): # Transl respect sc
            sym_list[isym,ll,3] = sym_list[isym,ll,3]/sc_size[ll]

    # Create an object from the Phonons class of the SC
    dyn_sc = dyn.GenerateSupercellDyn(sc_size)

    # Get the symmetries of the supercell
    spg_syms_sc = spglib.get_symmetry(dyn_sc.structure.get_spglib_cell(), symprec)
    sym_list_sc = cellconstructor.symmetries.GetSymmetriesFromSPGLIB(spg_syms_sc, regolarize= False)
    
    # Obtain symmetries that are pure translations:
    translation = np.zeros((Nsupercell,3,4),dtype=np.float64)
    for i in range(Nsupercell):
        translation[i,:,:] = sym_list_sc[i*Nsym] # The first is always a translation
    
    T_list = np.empty([sc_size[0]*sc_size[1]*sc_size[2], 3], dtype=np.float64)
    T_list_frac = np.empty([sc_size[0]*sc_size[1]*sc_size[2], 3], dtype=np.float64)
    for Tx in range(sc_size[0]):
        for Ty in range(sc_size[1]):
            for Tz in range(sc_size[2]):
                index = Tx*sc_size[2]*sc_size[1]+Ty*sc_size[2]+Tz
                T_list[index] = np.array([Tx, Ty, Tz])
                T_list_frac[index] = np.array([Tx/sc_size[0],Ty/sc_size[1],Tz/sc_size[2]])
    
    map_uc, map_tr = map_unitcell(dyn_sc, T_list_frac)

    #Obtain the rotations in cartesian coord:
    rot_cart = np.zeros((Nsym,3,3),dtype=np.float64)
    for isym in range(Nsym):
        rot_cart[isym] = cellconstructor.Methods.convert_matrix_cart_cryst2(sym_list[isym,:,:3], dyn.structure.unit_cell, cryst_to_cart = True)
        # Set elements smaller than the threshold to 0.0
        rot_cart[isym][np.abs(rot_cart[isym]) < 1e-12] = 0.0
    
    Natoms_sc = dyn_sc.structure.N_atoms
    
    singlet=np.zeros(1,dtype=np.intc)
    singlet_sym=np.zeros(1,dtype=np.intc)
    
    mapping = np.zeros((Natoms_sc, Nsym), dtype='<i4')
    orbit1s = np.zeros((Natoms_sc, Nsym), dtype='<i4')
    Nref_singlet=0
    for i in range(Natoms_sc):
        singlet = i
        Nref_singlet+=1
        Nequiv=0
        for isym in range(Nsym):
            sym_struct = dyn_sc.structure.copy()
            sym_struct.apply_symmetry(sym_list[isym],delete_original= True)
            # For each symmetry operation, we find which is equivalent to the i-th atom.
            irt = np.array(sym_struct.get_equivalent_atoms(dyn_sc.structure), dtype =np.intc)
            singlet_sym=irt[singlet]
            mapping[Nref_singlet-1, Nequiv] = singlet_sym
            orbit1s[Nref_singlet-1, Nequiv] = isym
            Nequiv+=1

    # return(mapping, rot_cart, map_uc, map_tr, T_list, T_list_frac)
    return mapping, map_uc, rot_cart

def map_unitcell(dyn_supercell, T_list):
    """
    Create a map of atomic indeces after the application of the Translation operation that bring atoms in the SC (ii) to the unit cell
    """
    ntot = dyn_supercell.structure.N_atoms
    nsupercell = T_list.shape[0]
    nat = ntot/nsupercell

    map_uc = np.zeros((ntot,ntot),dtype=np.intc)
    map_tr = np.zeros((ntot), dtype=np.intc)
    translation = np.empty([len(T_list),3,4], dtype=np.float64)
    for i in range(len(T_list)):
        translation[i,:,:3] = np.identity(3)
        translation[i,:,3] = T_list[i]

    for ii in range(ntot):
        uc_index=ii%nat
        #Search the corresponding translation
        for j in range(nsupercell):
            shifted_struct  = dyn_supercell.structure.copy()
            shifted_struct.apply_symmetry(translation[j],delete_original=True)
            irt_shifted = np.array(shifted_struct.get_equivalent_atoms(dyn_supercell.structure), dtype =np.intc)
            index_shifted = irt_shifted[ii]
            if (index_shifted == uc_index): # If after T, the first atom is in the 1uc:
                for jj in range(ntot):
                    map_uc[ii,jj] = irt_shifted[jj]
                    map_tr[ii] = j
                break
    return map_uc, map_tr

def map_q_to_1st_bz(rcell, q_cart, atol=1e-5):
    """
    Maps q_cart to the 1st BZ.
    On boundary ties (e.g., equidistant points), it deterministically
    prefers the vector that is lexicographically smaller in Cartesian coordinates.
    """
    q_cart = np.array(q_cart).flatten()
    rcell = np.array(rcell)

    inv_rcell = np.linalg.inv(rcell)
    q_frac = q_cart @ inv_rcell
    q_frac_wrapped = q_frac - np.round(q_frac)

    shifts = np.array([-1, 0, 1])
    grid_frac = np.stack(np.meshgrid(shifts, shifts, shifts), -1).reshape(-1, 3)

    candidate_fracs = q_frac_wrapped + grid_frac
    candidate_carts = candidate_fracs @ rcell

    distances_sq = np.sum(candidate_carts**2, axis=1)
    min_dist_sq = np.min(distances_sq)

    # This tolerance must be wider than the 6-digit precision of the input
    is_close = np.abs(distances_sq - min_dist_sq) < atol
    tied_candidates = candidate_carts[is_close]

    idx = np.lexsort((tied_candidates[:, 2], tied_candidates[:, 1], tied_candidates[:, 0]))

    return tied_candidates[idx[0]]

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
