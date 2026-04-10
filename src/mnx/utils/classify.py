import spglib
import numpy as np

import mnx.utils.cell as _cell


def convert_matrix_cart_cryst2(matrix, unit_cell, cryst_to_cart = False):
    """
    From CellConstructor.
    This methods convert the 3x3 matrix into crystalline coordinates using the metric tensor defined by the unit_cell.

    This perform the exact transform. 
    With this method you get a matrix that performs the transformation directly in the other space. 
    If I have a matrix that transforms vectors in crystalline coordinates, then with this I get the same operator between
    vectors in cartesian space.
    
    This subroutine transforms operators, while the previous one transforms matrices (obtained as open product between vectors)
    
    For example, use convert_matrix_cart_cryst to transform a dynamical matrix, 
    use convert_matrix_cart_cryst2 to transform a symmetry operation.
    
        
    Parameters
    ----------
        matrix : ndarray 3x3
            The matrix to be converted
        unit_cell : ndarray 3x3
            The cell containing the vectors defining the metric (the change between crystalline and cartesian coordinates)
        cryst_to_cart : bool, optional
            If False (default) the matrix is assumed in cartesian coordinates and converted to crystalline. If True
            otherwise.
            
    Results
    -------
        new_matrix : ndarray(3x3)
            The converted matrix into the desidered format
    """
    
    
    # Get the metric tensor from the unit_cell
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    # Choose which conversion perform
    comp_matrix = np.einsum("ij, jk", np.linalg.inv(metric_tensor), unit_cell) 
    comp_matrix_inv = np.linalg.inv(comp_matrix)
        
    if cryst_to_cart:
        return comp_matrix_inv.dot( np.dot(matrix, comp_matrix))

    return comp_matrix.dot( np.dot(matrix, comp_matrix_inv))

def GetIRT(structure, symmetry):
    """    
    """
    
    irt = np.empty([structure.Natoms], dtype=np.int64)

    for atom in range(structure.Natoms):
        new_coords_frac = symmetry[:,:3] @ _cell.cart2cryst(structure.atom_coords[atom,:], structure.cell)+symmetry[:,3]
        new_coords = _cell.cryst2cart(_to_unit_cell(new_coords_frac), structure.cell)
        i = np.where(np.isclose(new_coords,structure.atom_coords, atol=1e-3).all(axis=1))
        try:
            irt[atom] = i[0]
        except:
            breakpoint()
    return irt

def _to_unit_cell(coords_frac):
    # 6 ra aldatu degu.
    coords_frac[0] = np.round(coords_frac[0], 10)%1
    coords_frac[1] = np.round(coords_frac[1], 10)%1
    coords_frac[2] = np.round(coords_frac[2], 10)%1
    return coords_frac

def get_invs(QE_s, QE_nsym):
    """
    """
    QE_invs = np.zeros(48, dtype = np.intc)
    for i in range(QE_nsym):
        found = False
        for j in range(QE_nsym):
            if (QE_s[:,:,i].dot(QE_s[:,:,j]) == QE_s[:,:,0]).all():
                QE_invs[i] = j + 1 # Fortran index
                found = True
        
        if not found:
            raise ValueError
            
    return QE_invs

def GetSymmetriesFromSPGLIB(spglib_sym):
    """
    """
    
    # Check if the type is correct
    if not "translations" in spglib_sym:
        raise ValueError("Error, your symmetry dict has no 'translations' key.")
        
    if not "rotations" in spglib_sym:
        raise ValueError("Error, your symmetry dict has no 'rotations' key.")
    
    # Get the number of symmetries
    out_sym = []
    n_sym = np.shape(spglib_sym["translations"])[0]
    
    translations = spglib_sym["translations"]
    rotations = spglib_sym["rotations"]
    
    for i in range(n_sym):
        # Create the symmetry
        sym = np.zeros((3,4))
        sym[:,:3] = rotations[i, :, :]
        sym[:, 3] = translations[i,:]    
        out_sym.append(sym)
    
    return out_sym