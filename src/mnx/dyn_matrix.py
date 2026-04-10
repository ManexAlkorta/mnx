import mnx.utils.cell as _cell
import mnx.utils.consts as _consts
import mnx.utils.classify as _classify
import mnx.FModules

import symph

import numpy as np
import spglib
import copy

from .structure import Structure


class DynMatrix:
    """
    Main class which contains all the information of a phonon spectra. Its utilities are presented
    in the Tutorial/DynMatrix.ipynb notebook.
    """

    @classmethod
    def from_file(cls, file: str, qgrid: list=[0,0,0], Nqirr: int = 0) -> "DynMatrix": 
        """
        Classmethod of DynMatrix. Reads the phonon spectra from a set of QuantumEspresso
        dyn files.

        Parameters
        ----------
        file : str
            Path to dyn including the prefix. In the case of reading a unique q-point,
            the complete file name needs to be included.
        qgrid : list or np.ndarray
            qgrid where the phonon spectra is computed.
            Defaults to [0,0,0].
        Nqirr : int
            Number of irreducible q-points in the grid. If Nqirr=0, a unique q-point is 
            read, and file must be the complete file name.
            Defaults to [0,0,0].

        Returns
        -------
        DynMatrix : object
            An instance with the class containing the phonon spectra.
        """
        instance = cls()   
        instance.file = file
        instance.Nqirr = Nqirr
        if instance.Nqirr == 0:
            instance.structure = Structure.from_file(file, format="dyn")
        else:
            instance.qgrid = qgrid
            instance.structure = Structure.from_file(f"{file}1", format="dyn")
            instance.super_structure = instance.structure.expand_structure(instance.qgrid)
        if instance.Nqirr == 0:
            instance.DynQs = np.empty([1], dtype=object)
            instance.qstars = np.empty([1], dtype=object)

            instance.DynQs = DynQ.from_file(instance.file)
            instance.qstars[0] =instance.DynQs[0].qpoints
            instance.Nqpoints = instance.DynQs[qs].Nqinstar
        else:
            instance.DynQs = np.empty([instance.Nqirr], dtype=object)
            instance.qstars = np.empty([instance.Nqirr], dtype=object)
            for qirr in range(instance.Nqirr):
                instance.DynQs[qirr] = DynQ.from_file(f"{instance.file}{qirr+1}")
                instance.qstars[qirr] = instance.DynQs[qirr].qpoints
            instance.Nqpoints = 0
            for qs in range(instance.Nqirr):
                instance.Nqpoints += instance.DynQs[qs].Nqinstar
            instance.get_phiR()
        instance._alat = instance.DynQs[0]._alat
        instance.qgrid = qgrid
        return(instance)

    def distort_structure(self, Q : list, modes : list, mod : list=None) -> "Structure":
        """
        A function to generate a distorted structure from a given DynMatrix.

        Parameters
        ----------
        Q : list or np.ndarray
            The order parameter containing the combination for each of the independent 
            stars considered for the distortion. If a one dimensional star, and a 3 dimensional
            star are combined, the format is Q = [np.array([Q1]), np.array([Q2,Q3,Q4])].
        modes : list or np.ndarray
            Which modes/stars are considered for the distortion. First index corresponds to the 
            q-star, and second index to the vibrational mode. The format to consider the first mode
            of first q-star (1 dimensional), and the second mode of the second q-star (3-dimensional)
            is modes = [np.array([0,0]), np.array([1,1])]. Note that Q and mode must coincide in
            the degeneracy of the star and number of Qs.
        mod : list
            The size of the supercell for the output structure.
            Defaults to the size of the qgrid.

        Returns
        -------
        Structure : object
            Instance containing all the information about the crystalline structure.
        """

        if mod == None:
            mod = self.qgrid

        Q = np.asarray(Q)
        tmp_cell = np.empty([3, 3])
        for i in range(3):
            tmp_cell[i, :] = self.structure.cell[i, :] * mod[i]
        tmp_atom_coords = np.empty([self.structure.Natoms * mod[0] * mod[1] * mod[2], 3])
        tmp_atom_species = np.empty(
            [self.structure.Natoms * mod[0] * mod[1] * mod[2], 2], dtype="<U5"
        )
        for i in range(mod[0]):
            for j in range(mod[1]):
                for k in range(mod[2]):
                    tmp_atom_coords[
                        self.structure.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k) : self.structure.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k + 1)
                    ] = self.structure.atom_coords + (
                        self.structure.cell[0, :] * i
                        + self.structure.cell[1, :] * j
                        + self.structure.cell[2, :] * k
                    )
                    tmp_atom_species[
                        self.structure.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k) : self.structure.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k + 1)
                    ] = self.structure.atom_species
        tmp_disp_list = (self.expand_polvecs(modes=modes, mod=mod))[1]
        for qs, displacement_instar in enumerate(tmp_disp_list):
            for i, mode in enumerate(displacement_instar):
                rs_tmp_displ = np.reshape(mode, tmp_atom_coords.shape)
                tmp_atom_coords += np.real(Q[qs,i] * rs_tmp_displ)
        structure = Structure.from_data(tmp_cell, tmp_atom_coords, tmp_atom_species)
        return structure
    
    def info(self) -> None:
        """
        Returns the most relevant data of the DynMatrix object concisely.
        """
        self.structure.info()
        print("Dynamical information:")
        print(f"\tNumber of stars: {self.Nqirr}")
        for qs in range(self.Nqirr):
            for qi,q in enumerate(self.DynQs[qs].qpoints):
                if qi == 0:
                    print("\t\tStar {0:>3}: [{1:14.8f},{2:14.8f},{3:14.8f}]".format(qs,q[0],q[1],q[2]))
                else:
                    print("\t\t          [{1:14.8f},{2:14.8f},{3:14.8f}]".format(qs,q[0],q[1],q[2]))
            if qs != self.Nqirr-1:
                print("\n")
    
    def expand_polvecs(self, modes : list, mod : list) -> tuple[list, list, np.ndarray]:
        """
        Function to expand the polarization vectors out from the primitive cell.

        Parameters
        ----------
            modes: list or np.ndarray.
                Which polarization vectors or vibrational modes will be expanded.
                Example: [[2,0]], which means q-star 2, and vibrational mode 0.
            mod : list or np.ndarray.
                Modulation of the expanded polvecs. A typical choice would be the q-grid.
        Returns
        -------
            polvecs_list : list of np.array.
                List of np.arrays containing the expanded polarization vectors for the specified 
                modes in the q-star.
            displacements_list : list of np.array.
                List of np.arrays containing the displacemt vectors for the specified 
                modes in the q-star.
            exp_masses : np.array.
                Masses for each atom in the supercell speciefied by mod. 
        """
        exp_masses = np.empty([self.structure.Natoms * mod[0] * mod[1] * mod[2] * 3], complex)
        polvecs_list = np.empty(len(modes), dtype=object)
        displacements_list = np.empty(len(modes), dtype=object)
        for ii,index in enumerate(modes):
            polvecs_instar = np.empty([self.DynQs[index[0]].Nqinstar, self.structure.Natoms*mod[0]*mod[1]*mod[2]*3], dtype=complex)
            displacements_instar = np.empty([self.DynQs[index[0]].Nqinstar, self.structure.Natoms*mod[0]*mod[1]*mod[2]*3], dtype=complex)
            for qi,q in enumerate(self.DynQs[index[0]].qpoints):
                q = _cell.cryst2cart(q, self.structure.rcell)
                for i in range(mod[0]):
                    for j in range(mod[1]):
                        for k in range(mod[2]):
                            polvecs_instar[
                                qi,
                                self.structure.Natoms
                                * 3
                                * (mod[1] * mod[2] * i + mod[2] * j + k) : self.structure.Natoms
                                * 3
                                * (mod[1] * mod[2] * i + mod[2] * j + k + 1)
                            ] = self.DynQs[index[0]].polvecs[qi, index[1]] * np.exp(
                                2j
                                * np.pi
                                * np.matmul(
                                    _cell.cart2cryst(q, self.structure.rcell),
                                    np.array([i, j, k]),
                                )
                            )
                            for atom in range(self.structure.Natoms):
                                exp_masses[
                                    self.structure.Natoms
                                    * 3
                                    * (mod[1] * mod[2] * i + mod[2] * j + k)
                                    + atom
                                    * 3 : self.structure.Natoms
                                    * 3
                                    * (mod[1] * mod[2] * i + mod[2] * j + k)
                                    + (atom + 1) * 3
                                ] = self.structure.masses[atom]
                polvecs_instar[qi] = polvecs_instar[qi] / np.linalg.norm(polvecs_instar[qi])
                ##############################################################################
                ##### The displacements are constructed from already normalized pol vecs #####
                ##############################################################################
                for i in range(self.structure.Natoms * mod[0] * mod[1] * mod[2] * 3):
                    displacements_instar[qi,i] = polvecs_instar[qi,i] / np.sqrt(exp_masses[i])
            polvecs_list[ii] = polvecs_instar
            displacements_list[ii] = displacements_instar
        return polvecs_list, displacements_list, exp_masses

    def get_phiR(self) -> None:
        """
        Function to compute the real space force constants from the reciprocal space force constants. It 
        uses a the fortran subroutine mnx.Fmodules.interpolation.ift_fcq for efficiency.
        """
        phiqs = np.empty([self.Nqpoints,self.structure.Natoms,self.structure.Natoms,3,3], dtype=np.complex128)
        qpoints = np.empty([self.Nqpoints,3], dtype=np.float128)
        ii = 0
        for qs in range(self.Nqirr):
            for qi in range(self.DynQs[qs].Nqinstar):
                phiqs[ii] = self.DynQs[qs].phiqs[qi]
                qpoints[ii] = self.DynQs[qs].qpoints[qi]
                ii+=1
        self.phiR = np.real(mnx.FModules.interpolation.ift_fcq(phiqs,qpoints,self.structure.cell,self.super_structure.atom_coords,"lattice"))
    
    def get_dynq(self, q : list, gauge : str = "atomic") -> "DynQ":
        """
        Function to compute the reciprocal space force constants \Phi(Q). It employs the fortran
        subroutine mnx.FModules.interpolation.ft_fcr for efficiency.

        Parameters
        ----------
            q : list or np.ndarray.
                q-point in fractional units for which the \Phi(Q) is interpolated.
            gauge : str.
                Gauge employed in the interpolation. 
                Supported options (defaults to atomic):
                    atomic: atomic gauge is employed.
                    lattice: lattice gauge is employed.
        Returns
        -------
            dynq : DynQ object.
        """
        phiq, frequencies, polvecs = mnx.FModules.interpolation.ft_fcr(self.phiR,q,self.structure.masses,self.qgrid,self.structure.cell,self.super_structure.atom_coords,gauge)
        dynq = DynQ.from_data(self.structure, np.array([q]), np.array([phiq]), np.array([frequencies]), np.array([np.transpose(polvecs)]))
        return dynq
    
    def get_bands(self, k_inpath : np.ndarray, N : int) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Function to complute the phonon spectra in a fine grid.
        
        Parameters
        ----------
            k_inpath : np.ndarray
                k-points in fractional units, orginized in pairs defining paths.
                Example: np.array([[0,0,0],[0.5,0,0]],[[0.5,0,0],[0.5,0,0.5]]) corresponds to a
                GMA path in a hexagonal unit cell.
            N : int
                Number k-points in each of the segments defined in k_inpath.
        Returns
        -------
            bands : np.ndarray. [Nqpoints, Nmodes].
                Frequency of the vibrational modes in the q-path. Units cm^{-1}.
            cart_qpath : np.ndarray.
                Distance between respective q-points in cartesian units for a physical and straightforward
                visualization. 
            xticks : list.
                The values of cart_qpath, where the segements defined in k_inpath start/end.
        """
        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        bands = np.empty([len(k_inpath)*(N+1), self.structure.Natoms*3], dtype=complex)
        cart_qpath = np.zeros(len(k_inpath)*(N+1))
        xticks = [0]
        for ki,ks in enumerate(k_inpath):
            dk = (ks[1]-ks[0])
            cart_qpath[ki*(N+1):(ki+1)*(N+1)] = cart_qpath[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell))*np.arange(0,10+10/N,10/N)
            xticks.append(cart_qpath[(ki+1)*(N+1)-1])
            dk = dk
            qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]
        for qi,q in enumerate(qpath):
            dynq = self.get_dynq(q)
            for mode in range(self.structure.Natoms*3):
                bands[qi] = dynq.frequencies[0]
        return bands, cart_qpath, xticks
    
    def Symmetrize(self, symprec : float = 1e-5, apply_translations : bool = True) -> None:
        """
        Function that symmetrizes the force constants, and crystalline structure with the precision
        specified by symprec.

        Parameters
        ----------
            symprec : float.
                Precision used to identify the symmetry operations with spglib. Check Structure.get_spacegroup()
                to select the desired resulting space-group.
                Defaults to 1e-5.
            apply_translation : bool.
                If pure translations will be considered in the symmetrization.
                Defaults to True.
        """
        self.structure.Symmetrize(symprec)
        self.super_structure = self.structure.expand_structure(self.qgrid)
        self.SetupFromSPGLIB(symprec=symprec)
        tmp_phiR = np.empty([self.super_structure.Natoms*3, self.super_structure.Natoms*3])
        for a in range(self.super_structure.Natoms):
            for b in range(self.super_structure.Natoms):
                tmp_phiR[a*3:(a+1)*3,b*3:(b+1)*3] = self.phiR[a,b,:,:]

        # Apply the Permutation symmetry
        tmp_phiR[:,:] = 0.5 * (tmp_phiR + tmp_phiR.T)
        # First lets recall that the fortran subroutines
        # Takes the input as (3,3,nat,nat)
        new_phiR = np.zeros( (3,3, self._QE_nat, self._QE_nat), dtype = np.double, order ="F")
        for i in range(self._QE_nat):
            for j in range(self._QE_nat):
                new_phiR[:, :, i, j] = tmp_phiR[3*i : 3*(i+1), 3*j : 3*(j+1)]
        # Apply the translations
        if apply_translations:
            # Check that the translations have been setted up
            assert len(np.shape(self._QE_translations_irt)) == 2, "Error, symmetries not setted up to work in the supercell"
            symph.trans_v2(new_phiR, self._QE_translations_irt)
        # Apply the symmetrization
        symph.sym_v2(new_phiR, self._QE_at, self._QE_bg, self._QE_s, self._QE_irt, self._QE_nsym, self._QE_nat)
        # Return back
        for i in range(self._QE_nat):
            for j in range(self._QE_nat):
                tmp_phiR[3*i : 3*(i+1), 3*j : 3*(j+1)] = new_phiR[:, :, i, j]

        for a in range(self.super_structure.Natoms):
            for b in range(self.super_structure.Natoms):
                self.phiR[a,b,:,:] = tmp_phiR[a*3:(a+1)*3,b*3:(b+1)*3]
        for si,star in enumerate(self.qstars):
            for qi,q in enumerate(star):
                phiq, frequencies, polvecs = mnx.FModules.interpolation.ft_fcr(self.phiR,q,self.structure.masses,self.qgrid,self.structure.cell,self.super_structure.atom_coords,"lattice")
                dynq = DynQ.from_data(self.structure, np.array(self.qstars[qi]), np.array([phiq]), np.array([frequencies]), np.array([np.transpose(polvecs)]))
                self.DynQs[si].phiqs[qi] = dynq.phiqs[0]
                self.DynQs[si].dynqs[qi] = dynq.dynqs[0]
                self.DynQs[si].frequencies[qi] = dynq.frequencies[0]
                self.DynQs[si].polvecs[qi] = dynq.polvecs[0]
                self.DynQs[si].displacements[qi] = dynq.displacements[0]
        
    def SetupFromSPGLIB(self, symprec : float) -> None:
        """
        Prepares the DynMatrix objects auxiliary parameters self._QE... for symmetrization of
        force-constants via QuantumESPRESSO's symmetrization module.

        Parameters
        ----------
            symprec : float
                Precision used to identify the symmetry operations with spglib. Check Structure.get_spacegroup()
                to select the desired resulting space-group.
         """
        # Get the symmetries
        spg_syms = spglib.get_symmetry(self.super_structure.get_spglib_cell(), symprec)
        symmetries = _classify.GetSymmetriesFromSPGLIB(spg_syms)

        self._QE_nat = self.super_structure.Natoms
        self._QE_s = np.zeros( (3, 3, 48) , dtype = np.intc, order = "F")
        self._QE_s_cart = np.zeros( (3, 3, 48) , dtype = np.float64, order = "F")
        self._QE_s_inv_cart = np.zeros( (3, 3, 48) , dtype = np.float64, order = "F")
        self._QE_irt = np.zeros((48, self.super_structure.Natoms), dtype = np.intc, order = "F")
        self._QE_invs = np.zeros( (48), dtype = np.intc, order = "F")
        self._QE_at = np.zeros( (3,3), dtype = np.float64, order = "F")
        self._QE_bg = np.zeros( (3,3), dtype = np.float64, order = "F")

        bg = self.super_structure.rcell
        for i in range(3):
            for j in range(3):
                self._QE_at[i,j] = self.super_structure.cell[j,i]
                self._QE_bg[i,j] = bg[j,i]

        # Check how many point group symmetries do we have
        n_syms = 0
        for i, sym in enumerate(symmetries):
            # Extract the rotation and the fractional translation
            rot = sym[:,:3]
            rot_cart = _classify.convert_matrix_cart_cryst2(rot, self.super_structure.cell, cryst_to_cart = True)
            # Check if the rotation is equal to the first one
            if np.sum( (rot - symmetries[0][:,:3])**2 ) < 0.1 and n_syms == 0 and i > 0:
                # We got all the rotations
                n_syms = i 
                break
                
            # Extract the point group
            if n_syms == 0:
                self._QE_s[:,:, i] = rot.T
                self._QE_s_cart[:,:, i] = rot_cart
                self._QE_s_inv_cart[:,:, i] = rot_cart.T
                # Get the IRT (Atoms mapping using symmetries)
                irt = _classify.GetIRT(self.super_structure, sym)
                self._QE_irt[i, :] = irt + 1 #Py to Fort
        
        if n_syms == 0:
            n_syms = len(symmetries)
        
        # From the point group symmetries, get the supercell
        n_supercell = len(symmetries) // n_syms
        self._QE_translation_nr = n_supercell
        self._QE_nsymq = n_syms
        self._QE_nsym = n_syms

        self._QE_translations_irt = np.zeros((self.super_structure.Natoms, self.Nqpoints), dtype = np.intc, order = "F")
        self._QE_translations = np.zeros((3, self.Nqpoints), dtype = np.double, order = "F")


        # Now extract the translations
        for i in range(self.Nqpoints):
            sym = symmetries[i * n_syms]
            # Check if the symmetries are correctly setup

            I = np.eye(3)
            ERROR_MSG="""
            Error, symmetries are not correctly ordered.
            They must always start with the identity.

            N_syms = {}; N = {}; SYM = {}
            """.format(n_syms,i*n_syms, sym)
            assert np.sum( (I - sym[:,:3])**2) < 0.5, ERROR_MSG

            # Get the irt for the translation (and the translation)
            irt = _classify.GetIRT(self.super_structure, sym)
            self._QE_translations_irt[:, i] = irt + 1
            self._QE_translations[:, i] = sym[:,3]

        # For each symmetry operation, assign the inverse
        self._QE_invs[:] = _classify.get_invs(self._QE_s, self._QE_nsym)

    def change_cell(self, cell : np.ndarray) -> None:
        """
        This function changes the cell manually, moving the atomic coordinates by fixing the fractional coordinates.

        Parameters
        ----------
            cell : np.ndarray. [3,3].
                New unit-cell.
        """
        self.structure.atom_coords = _cell.cryst2cart(_cell.cart2cryst(self.structure.atom_coords,self.structure.cell),cell)
        self.structure.cell = cell
        self.structure.rcell = _cell.get_rcell(self.structure.cell)
        self._alat = np.linalg.norm(self.structure.cell[0,:])/_consts.bohr2angstroms
        for dynq in self.DynQs:
            dynq.structure = self.structure
            dynq._alat = self._alat

    def copy(self) -> "DynMatrix":
        """
        Generates a copy of the DynMatrix object.
        """
        return copy.deepcopy(self)

    def write(self, file : str) -> None:
        """
        This function writes each of the DynQs in QuantumESPRESSO format.
        """
        for si, star in enumerate(self.qstars):
            self.DynQs[si].write(f"{file}{si+1}")


class DynQ:
    """
    Subclass of DynMatrix, containing info related to an individual q-star at reciprocal
    space.
    """
    @classmethod
    def from_file(cls, file : str) -> "DynQ":
        """
        Classmethod of DynQ subclass. Reads the dynamical matrix of the q-star from an individual
        QuantumESPRESSO dyn file.

        Parameters
        ----------
            file : str
                Name of the espresso dyn matrix.
        """
        instance = cls()
        instance.structure = Structure.from_file(file, format="dyn")
        data = (open(file, "r")).readlines()
        instance._alat = float((data[2].split())[3]) * _consts.bohr2angstroms
        data1 = []
        for i in range(len(data)):
            try:
                if data[i].split()[0] != "\n":
                    data1.append(data[i])
            except:
                None
        data = data1
        instance.Nqinstar = -1
        for i in range(len(data)):
            if data[i].split()[0] == "q":
                instance.Nqinstar += 1
        instance.qpoints = np.empty([instance.Nqinstar, 3], dtype=float)
        instance.qpoints_alat = np.empty([instance.Nqinstar, 3], dtype=float)
        i = 3
        instance.phiqs = np.empty(
            [instance.Nqinstar, instance.structure.Natoms, instance.structure.Natoms, 3, 3], dtype=complex
        )
        for q_index in range(instance.Nqinstar):
            while (data[i][:]).split()[0] != "q":
                i += 1
            instance.qpoints[q_index] = np.array([(data[i][:]).split()[3:6]])
            instance.qpoints[q_index] = _cell.cart2cryst(instance.qpoints[q_index], instance.structure.rcell)
            instance.qpoints_alat[q_index] = instance.qpoints[q_index]/instance._alat
            i += 2
            for n1 in range(instance.structure.Natoms):
                for n2 in range(instance.structure.Natoms):
                    for n3 in range(3):
                        instance.phiqs[q_index, n1, n2, n3, 0] = complex(
                            float((data[i + n3].split())[0]),
                            float((data[i + n3].split())[1]),
                        )
                        instance.phiqs[q_index, n1, n2, n3, 1] = complex(
                            float((data[i + n3].split())[2]),
                            float((data[i + n3].split())[3]),
                        )
                        instance.phiqs[q_index, n1, n2, n3, 2] = complex(
                            float((data[i + n3].split())[4]),
                            float((data[i + n3].split())[5]),
                        )
                    i += 4
        instance.polvecs = np.empty(
            [instance.Nqinstar, instance.structure.Natoms * 3, instance.structure.Natoms * 3], complex
        )
        instance.frequencies = np.empty([instance.Nqinstar, instance.structure.Natoms * 3], complex)
        instance.displacements = np.empty(
            [instance.Nqinstar, instance.structure.Natoms * 3, instance.structure.Natoms * 3], complex
        )
        instance.dynqs = instance._phis2dyns()
        for qi in range(instance.Nqinstar):
            instance.frequencies[qi,:], instance.polvecs[qi,:], instance.displacements[qi,:] = instance._diagdynq(qi)
        
        instance.frequencies = np.sqrt(instance.frequencies) * _consts.Ry2cm
        return instance
    
    @classmethod
    def from_data(cls, structure : Structure, qpoints : np.ndarray, phiqs : np.ndarray, frequencies : np.ndarray, polvecs : np.ndarray) -> "DynQ":
        """
        Classmethod of DynQ. Generates a DynQ object from data.

        Parameters
        ----------
            structure : Structure object.
                Structure of the system.
            qpoints : np.ndarray. [Nqinstar, 3].
                q-points in the star. Fractional units.
            phiqs : np.ndarray. [Nqinstar, Natoms, Natoms, 3, 3].
                Force constants at reciprocal space. Espresso units.
            frequencies : np.ndarray. [Nqinstar, Natoms*3].
                Eigenvalues of the dynamical matrix for each of the q-points in the star. Units cm^{-1}.
            polvecs : np.ndarray. [Nqinstar, Natoms*3, Natoms*3].
                Eigenvectros of the dynamical matrix for each of the q-points in the star. The first
                dimension corresonds to the q-point, the second to the vibrational mode, and the third
                to the combined a:atom, alpha:cart_index superindex.
        """
        instance = cls()
        instance.structure = structure
        instance.qpoints = qpoints
        instance.Nqinstar = qpoints.shape[0]
        instance.phiqs = phiqs
        instance.dynqs = instance._phis2dyns()
        instance.polvecs = np.empty([instance.Nqinstar,instance.structure.Natoms*3,instance.structure.Natoms*3], dtype=np.complex128)
        instance.displacements = np.empty([instance.Nqinstar,instance.structure.Natoms*3,instance.structure.Natoms*3], dtype=np.complex128)
        instance.frequencies = np.empty([instance.Nqinstar,instance.structure.Natoms*3], dtype=np.complex128)

        for qi in range(instance.Nqinstar):
            for a in range(structure.Natoms):
                instance.displacements[qi,:] = polvecs[qi]/np.sqrt(instance.structure.masses[a])
            instance.frequencies[qi,:], instance.polvecs[qi,:] = frequencies[qi], polvecs[qi]
        
        instance.frequencies = np.sqrt(instance.frequencies) * _consts.Ry2cm
        return instance

    def _phis2dyns(self) -> None:
        """
        This function mass normalizes the force contants to obtain the real dynamical matrix, which
        is later diagonalized.
        """
        dynqs = np.empty(
            [self.Nqinstar, self.structure.Natoms, self.structure.Natoms, 3, 3], dtype=complex
        )
        for qi, phiq in enumerate(self.phiqs):
            for n1 in range(self.structure.Natoms):
                for n2 in range(self.structure.Natoms):
                    dynqs[qi, n1, n2, :, :] = phiq[n1, n2, :, :] / np.sqrt(self.structure.masses[n1] * self.structure.masses[n2])
        return dynqs

    def _diagdynq(self, qi : int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function diagonalizes the dynamical matrix, and orders from low tho high frequency the 
        vibrational spectra.
        
        Parameters
        ----------
            qi : int.
                The q-id for the q-point in the star to be diagonalized.
        Returns
        -------
            frequencies : np.ndarray. [Natoms*3]
                Eigenvalues of the dynamical matrix.
            polvecs : np.ndarray. [Natoms*3, Natoms*3]
                Eigenvectors of the dynamical matrix.
            displacements : np.ndarray.
                Mass renormalized eigenvalues of the dynamical matrix.
        """
        D = np.empty([self.structure.Natoms*3, self.structure.Natoms*3], dtype=complex)
        for i in range(self.structure.Natoms*3):
            for j in range(self.structure.Natoms*3):
                n1 = i // 3
                n2 = j // 3
                n3 = i % 3
                n4 = j % 3
                D[i, j] = self.dynqs[qi,n1, n2, n3, n4]
        frequencies, polvecs = np.linalg.eig(D)
        displacements = np.empty(polvecs.shape, dtype = np.complex128)
        for mode in range(self.structure.Natoms * 3):
            for atom in range(self.structure.Natoms):
                i = 3 * atom
                displacements[mode, i:i+3] = polvecs[mode, i:i+3] / np.sqrt(self.structure.masses[atom])
        frequencies, polvecs, displacements = self._reorder(frequencies, polvecs, displacements)
        return (frequencies, polvecs, displacements)
    
    def _reorder(self, tmp_frequencies : np.ndarray, tmp_polvecs : np.ndarray, tmp_displacements : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function reorders the phonon spectra from low to high frequencies.

        Parameters
        ----------
            tmp_frequencies : np.ndarray. [Natoms*3]
                Eigenvalues of the dynamical matrix.
            tmp_polvecs : np.ndarray. [Natoms*3, Natoms*3]
                Eigenvectors of the dynamical matrix.
            tmp_displacements : np.ndarray.
                Mass renormalized eigenvalues of the dynamical matrix.
        Returns
        -------
            frequencies : np.ndarray. [Natoms*3]
                Reordered eigenvalues of the dynamical matrix.
            polvecs : np.ndarray. [Natoms*3, Natoms*3]
                Reordered eigenvectors of the dynamical matrix.
            displacements : np.ndarray.
                Reordered mass renormalized eigenvalues of the dynamical matrix.
        """
        tmp_frequencies = tmp_frequencies.transpose()
        tmp_polvecs = tmp_polvecs.transpose()
        tmp_displacements = np.empty(tmp_polvecs.shape, dtype = np.complex128)
        for mode in range(self.structure.Natoms * 3):
            for atom in range(self.structure.Natoms):
                i = 3 * atom
                tmp_displacements[mode, i:i+3] = tmp_polvecs[mode, i:i+3] / np.sqrt(self.structure.masses[atom])
        frequencies = np.empty(tmp_frequencies.shape, complex)
        polvecs = np.empty(tmp_polvecs.shape, complex)
        displacements = np.empty(tmp_polvecs.shape, complex)
        order = np.argsort(tmp_frequencies)
        for i, n in enumerate(order):
            frequencies[i] = tmp_frequencies[n]
            polvecs[i] = tmp_polvecs[n]
            displacements[i] = displacements[n]
        return frequencies, polvecs, displacements
    
    def copy(self) -> "DynQ":
        """
        This function makes a copy of the DynQ object.
        """
        return copy.deepcopy(self)
    
    def write(self, file : str, alat : bool = True) -> None:
        """
        This function writes the DynQ object in QuantumESPRESSO format.
        
        Parameters
        ----------
            file: str.
                Name of the file to be written.
            alat: bool.
                If we want to renormalize the dynamical matrix in alat units as ESPRESSO does. Usefull
                is the dyn matrix will be used in a QunatumESPRESSO calcuation in the later.
        """
        file = open(file, "w")
        file.write(f"Dynamical matrix file\n\n")
        fmt_general_info = "{0:>3d}{1:>5d}{2:>4d}{3:14.7f}{4:14.7f}{5:14.7f}{6:14.7f}{7:14.7f}{8:14.7f}\n"
        fmt_cell = "  {0:15.9f}{1:15.9f}{2:15.9f}\n"
        fmt_masses = " {0:>12d}  '{1:<4}'{2:20.9f}\n"
        fmt_atomic_position = "{0:>5d}{1:>5d} {2:16.10f} {3:16.10f} {4:16.10f}\n"
        fmt_dyn_matrix="{0:12.8f} {1:12.8f}   {2:12.8f} {3:12.8f}   {4:12.8f} {5:12.8f}\n"
        if alat:
            norm = np.linalg.norm(self.structure.cell[0,:])
        else:
            norm = 1
        file.write(fmt_general_info.format(
            self.structure.Nspecies,
            self.structure.Natoms, 
            int(0),
            norm*_consts.angstroms2bohr,
            0,0,0,0,0)
            )
        file.write("Basis vectors\n")
        for i in range(3):
            file.write(fmt_cell.format(
                self.structure.cell[i, 0]/norm,
                self.structure.cell[i, 1]/norm,
                self.structure.cell[i, 2]/norm,
            ))
        name_list = []
        mass_list = []
        for atom in range(self.structure.Natoms):
            if self.structure.atom_species[atom, 0] not in name_list:
                name_list.append(self.structure.atom_species[atom,0])
                mass_list.append(self.structure.masses[atom])
        for s in range(self.structure.Nspecies):
            file.write(fmt_masses.format(
                int(s+1),
                name_list[s],
                mass_list[s]
            ))
        for atom in range(self.structure.Natoms):
            file.write(fmt_atomic_position.format(
                int(atom+1),
                int(self.structure.atom_species[atom, 1]),
                self.structure.atom_coords[atom, 0]/norm,
                self.structure.atom_coords[atom, 1]/norm,
                self.structure.atom_coords[atom, 2]/norm
            ))
        for qi in range(self.Nqinstar):
            file.write("\n     Dynamical  Matrix in cartesian axes\n\n")
            file.write("     q = ( {0:14.9f}{1:14.9f}{2:14.9f} )\n\n".format(
                self.qpoints_alat[qi, 0]*norm,
                self.qpoints_alat[qi, 1]*norm,
                self.qpoints_alat[qi, 2]*norm
            ))
            for i in range(self.structure.Natoms):
                for j in range(self.structure.Natoms):
                    file.write("{0:>5d}{1:>5d}\n".format(i+1, j+1))
                    for k in range(3):
                        file.write(fmt_dyn_matrix.format(
                            self.phiqs[qi, i, j, k, 0].real,
                            self.phiqs[qi, i, j, k, 0].imag,
                            self.phiqs[qi, i, j, k, 1].real,
                            self.phiqs[qi, i, j, k, 1].imag,
                            self.phiqs[qi, i, j, k, 2].real,
                            self.phiqs[qi, i, j, k, 2].imag,

                        ))
        file.write("\n     Diagonalizing the dynamical matrix\n\n")
        file.write("     q = ( {0:14.9f}{1:14.9f}{2:14.9f} )\n\n".format(
                self.qpoints_alat[0, 0]*norm, 
                self.qpoints_alat[0, 1]*norm,
                self.qpoints_alat[0, 2]*norm
            ))
        file.write(" **************************************************************************\n")
        for mode in range(self.structure.Natoms*3):
            if np.square(self.frequencies[0, mode]) < 0: ## Maybe this is the problem.
            #if np.abs(self.frequencies[0, mode].imag) > np.abs(self.frequencies[0, mode].real):
                file.write("     freq ({0:>4d}) = {1:14.6f} [THz] = {2:14.6f} [cm-1]\n".format(
                int(mode+1),
                np.abs(self.frequencies[0, mode].imag) * _consts.cm2Thz*(-1),
                np.abs(self.frequencies[0, mode].imag)*(-1)
            ))
            else:
                file.write("     freq ({0:>4d}) = {1:14.6f} [THz] = {2:14.6f} [cm-1]\n".format(
                    int(mode+1),
                    self.frequencies[0, mode].real * _consts.cm2Thz,
                    self.frequencies[0, mode].real
                ))
            for atom in range(self.structure.Natoms):
                disp = self.displacements[0, mode, :]/np.linalg.norm(self.displacements[0, mode, :])
                file.write("({0:10.6f}{1:10.6f}{2:10.6f}{3:10.6f}{4:10.6f}{5:10.6f} )\n".format(
                    disp[atom*3].real,
                    disp[atom*3].imag,
                    disp[atom*3+1].real,
                    disp[atom*3+1].imag,
                    disp[atom*3+2].real,
                    disp[atom*3+2].imag,
                ))
        file.write(" **************************************************************************\n")