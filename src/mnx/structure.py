import mnx.utils.cell as _cell
import mnx.utils.consts as _consts
import mnx.utils.classify as _classify

import numpy as np
import spglib
import ase.io
import copy


class Structure:
    """
    This function handles all the information and tools related with the crystal structure
    of a system. 
    """  
    @classmethod
    def from_file(cls, file : str, format : str) -> "Structure":
        """
        Classmethod of the Structure object. Reads an structure from a file, supporting QuantumESPRESSO
        scf and dyn matrix formats.

        Parameters
        ----------
            file : str.
                Name of the file to be read.
            format : str.
                Format of the file to be read.
                Supported formats:
                    - qe: QuantumESPRESSO scf input.
                    - dyn: QunatumESPRESSO dynamical matrix.
        Returns
        -------
            Structure : mnx.Structure object.
        """
        instance = cls()
        if format == "qe":
            data = (open(file, "r")).readlines()
            atomic_species_dict = {}
            instance.cell = np.empty([3, 3], dtype=float)
            i = 0
            while data[i][:15] != "CELL_PARAMETERS":
                i += 1
            instance.cell_units = data[i].split()[-1]
            i += 1
            for j in range(3):
                for k in range(3):
                    instance.cell[j, k] = float((data[i + j].split())[k])
            i += 3
            instance.rcell = _cell.get_rcell(instance.cell)
            while data[i][:16] != "ATOMIC_POSITIONS":
                i += 1
            i += 1
            instance.Natoms = len(data) - i 
            instance.atom_species = np.empty([instance.Natoms, 2], dtype="<U5")
            instance.atom_coords = np.empty([instance.Natoms, 3], dtype=float)
            tmp_id = 0
            for j in range(i, instance.Natoms + i):
                atom = j - i
                instance.atom_species[atom, 0] = (data[j].split())[0]
                if instance.atom_species[atom, 0] not in list(
                    atomic_species_dict.keys()
                ):
                    atomic_species_dict[instance.atom_species[atom, 0]] = tmp_id
                    tmp_id += 1
                instance.atom_species[atom, 1] = atomic_species_dict[
                    instance.atom_species[atom, 0]
                ]
                for k in range(3):
                    instance.atom_coords[atom, k] = (data[j].split())[k + 1]
            instance._atomic_dict = atomic_species_dict
            instance.Nspecies = tmp_id            
        elif format == "vasp":
            #Still testing
            tmp_ase_struct = ase.io.read(file,format="vasp")
            instance = instance.from_ASE(atom = tmp_ase_struct)
        elif format=="dyn":
            data = (open(file, "r")).readlines()
            instance.Nspecies = int((data[2].split())[0])
            instance.Natoms = int((data[2].split())[1])
            alat = float((data[2].split())[3]) * _consts.bohr2angstroms
            tmp_atom_coords = np.empty([instance.Natoms, 3])
            tmp_cell = np.empty([3, 3])
            instance.masses = np.empty([instance.Natoms])
            atomic_species_dict = {}
            atomic_masses_dict = {}
            tmp_atom_species = np.empty([instance.Natoms, 2], dtype="<U5")
            for i in range(3):
                for j in range(3):
                    tmp_cell[i, j] = float((data[4 + i].split())[j])
            instance.cell = alat * tmp_cell
            instance.rcell = _cell.get_rcell(tmp_cell)  # Not in alat units in Angstroms
            for i in range(instance.Nspecies):
                atomic_species_dict[data[7 + i].split()[1][1:]] = data[7 + i].split()[0]
                atomic_masses_dict[data[7 + i].split()[0]] = data[7 + i].split()[3]
            for atom in range(instance.Natoms):
                tmp_atom_coords[atom, 0] = float(data[7 + instance.Nspecies + atom].split()[2])
                tmp_atom_coords[atom, 1] = float(data[7 + instance.Nspecies + atom].split()[3])
                tmp_atom_coords[atom, 2] = float(data[7 + instance.Nspecies + atom].split()[4])
                tmp_atom_species[atom, 1] = data[7 + instance.Nspecies + atom].split()[1]
                key_list = list(atomic_species_dict.keys())
                val_list = list(atomic_species_dict.values())
                pos = val_list.index(tmp_atom_species[atom, 1])
                tmp_atom_species[atom, 0] = key_list[pos]
                instance.masses[atom] = atomic_masses_dict[tmp_atom_species[atom, 1]]
            instance._atomic_dict = atomic_species_dict
            instance.atom_coords = tmp_atom_coords * alat
            instance._set_atom_species(tmp_atom_species)
            instance._fix_coords()
        return instance
            
    @classmethod
    def from_ASE(cls, atom : ase.atom) -> "Structure":
        """
        Classmethod of the Structure object.Reads an ase.atom object, and translates
        the structure to an mnx.Structure object.

        Parameters
        ----------
            atom : ase.atom.
                Ase.atom object.
        Returns
        -------
            Structure : mnx.Structure object.
        """
        instance = cls()
        instance.cell = np.array(atom.get_cell())
        instance.atom_coords = np.array(atom.get_positions())
        instance.Natoms = atom.get_global_number_of_atoms()
        instance.atom_species = np.empty([instance.Natoms, 2], dtype="<U5")
        atomic_species_dict, k = {}, 1
        for a in range(instance.Natoms):
            if atom.symbols[a] not in instance.atom_species[:, 0]:
                atomic_species_dict[atom.symbols[a]] = k
                k += 1
            instance.atom_species[a, 0] = atom.symbols[a]
            instance.atom_species[a, 1] = atomic_species_dict[
                atom.symbols[a]
            ]
        instance._atomic_dict = atomic_species_dict
        list_species = []
        for a in range(instance.Natoms):
            if instance.atom_species[a, 0] not in list_species:
                list_species.append(instance.atom_species[a, 0])
        instance.Nspecies = len(list_species)
        instance.rcell = _cell.get_rcell(instance.cell)
        instance._fix_coords()
        return instance

    @classmethod
    def from_data(cls, cell: np.ndarray, atom_coords : np.ndarray, atom_species : np.ndarray) -> "Structure":
        """
        Classmethod of the Structure object. Reads the info from data.
        
        Parameters
        ----------
            cell: np.ndarray. [3,3].
                Cell in angstroms.
            atom_coords: np.ndarray. [Natoms, 3].
                Atomic coordinates in angstroms. 
            atom_species: np.ndarray. [Natoms, 2].
                Atomic species. First column corresponds to the atomic labels, and 
                the second to atomic ids.
        Returns
        -------
            Structure : Structure object.
        """
        instance = cls()
        instance.cell=cell
        instance.rcell = _cell.get_rcell(instance.cell)
        instance.atom_coords = atom_coords
        instance._set_atom_species(atom_species)
        instance._fix_coords()
        return instance

    def info(self) -> None:
        """
        This function prints the info about the Structure object.
        """
        print("Lattice information:")
        print(f"\tNumber of atoms: {self.Natoms}")
        print(f"\tNumber of atomic species: {self.Nspecies}")
        print(f"\tSpace group: {self.get_spacegroup()}")

        print("\tCell = [{0:14.8f},{1:14.8f},{2:14.8f}],". format(self.cell[0,0],self.cell[0,1],self.cell[0,2]))
        print("\t       [{0:14.8f},{1:14.8f},{2:14.8f}],". format(self.cell[1,0],self.cell[1,1],self.cell[1,2]))
        print("\t       [{0:14.8f},{1:14.8f},{2:14.8f}],". format(self.cell[2,0],self.cell[2,1],self.cell[2,2]))
        
        print(f"\tAtomistic structure:")
        for atom in range(self.Natoms):
            print("\t{0:>10}{1:14.8f},{2:14.8f},{3:14.8f}".format(self.atom_species[atom,0],self.atom_coords[atom,0],self.atom_coords[atom,1],self.atom_coords[atom,2]))


    def get_spacegroup(self, symprec : float = 1e-5) -> str:
        """
        Spglib function to calculate to which crystallographic spacegroup that 
        the Structure object belongs.
        
        Parameters
        ----------
            symprec: float. Defaults to 1e-5.
                Precision in the detection of symmetries.
        Retruns
        -------
            sg: str
                Crystallographic space-group.
        """
        sg = spglib.get_spacegroup(self.get_spglib_cell(), symprec=symprec)
        return sg

    def _set_atom_species(self, atom_species: np.ndarray) -> None:
        """
        Funtion to set the atomic species info of the Structure object.

        Parameters
        ----------
            atom_species: np.ndarray. [Natoms, 2].
                Atomic species. First column corresponds to the atomic labels, and 
                the second to atomic ids.
        """
        self.atom_species = atom_species
        self.Natoms = len(atom_species[:, 0])
        id_list = []
        for atom in range(self.Natoms):
            if self.atom_species[atom, 1] not in id_list:
                id_list.append(self.atom_species[atom, 1])
        self.Nspecies = len(id_list)


    def write(self, file : str, format : str) -> None:
        """
        Function to write the Structure object into file.

        Parameters
        ----------
            file: str.
                Name of the output file.
            format: str.
                Format of the output file. 
                Supported formats:
                    - qe: Outputs a structure in the QuantumESPRESSO format.
                    - vasp: Outputs a structure in the VASP POSCAR format.
        """
        if format == "qe":
            fmt_cell = "{0:18.9f}{1:18.9f}{2:18.9f}\n"
            fmt_atom_struct = "{0:>10}{1:18.9f}{2:18.9f}{3:18.9f}\n"
            file = open(file, "w")
            file.write("ATOMIC_POSITIONS {crystal}\n")
            atom_coords_crystal = _cell.cart2cryst(self.atom_coords, self.cell)
            for atom in range(self.Natoms):
                file.write(
                    fmt_atom_struct.format(
                        self.atom_species[atom, 0],
                        atom_coords_crystal[atom, 0],
                        atom_coords_crystal[atom, 1],
                        atom_coords_crystal[atom, 2],
                    )
                )
            file.write("\nCELL_PARAMETERS {" + self.cell_units + "}\n")
            for k in range(3):
                file.write(
                    fmt_cell.format(self.cell[k, 0], self.cell[k, 1], self.cell[k, 2])
                )
        elif format == "vasp":
            fmt_chemical_species = "{0:>4}"
            fmt_cell = "{0:18.9f}{1:18.9f}{2:18.9f}\n"
            fmt_atom_struct = "{0:18.9f}{1:18.9f}{2:18.9f} {3:>4}\n"
            file = open(file, "w")
            file.write("Title\n")
            file.write("{0:18.9f}\n".format(1.0))
            for i in range(3):
                file.write(
                    fmt_cell.format(self.cell[i, 0], self.cell[i, 1], self.cell[i, 2])
                )
            list_species = []
            list_species_kont = []
            for atom in range(self.Natoms):
                if self.atom_species[atom, 0] not in list_species:
                    file.write(fmt_chemical_species.format(self.atom_species[atom, 0]))
                    list_species.append(self.atom_species[atom, 0])
                    list_species_kont.append(1)
                else:
                    for i in range(len(list_species)):
                        if list_species[i] == self.atom_species[atom, 0]:
                            list_species_kont[i] = list_species_kont[i] + 1
            file.write("\n")
            for i in range(self.Nspecies):
                file.write(fmt_chemical_species.format(list_species_kont[i]))

            file.write("\nCartesian\n")
            for specie in range(self.Nspecies):
                for atom in range(self.Natoms):
                    if self.atom_species[atom, 0] == list_species[specie]:
                        file.write(
                            fmt_atom_struct.format(
                                self.atom_coords[atom, 0],
                                self.atom_coords[atom, 1],
                                self.atom_coords[atom, 2],
                                self.atom_species[atom, 0],
                            )
                        )

    def expand_structure(self, mod : list) -> "Structure":
        """
        This function expands the Structure out from the unit cell, to a supercell defined
        by the mod parameter.
        
        Parameters
        ----------
            mod: list.
                Supercell to which the output structure is expanded.
        Returns
        -------
            Structure: object.
                Structure object in a expanded supercell.
        """
        tmp_atom_coords = np.empty([self.Natoms*mod[0]*mod[1]*mod[2], 3], dtype=float)
        tmp_cell = np.empty([3, 3])
        for i in range(3):
            tmp_cell[i, :] = self.cell[i, :] * mod[i]
        tmp_atom_species = np.empty(
            [self.Natoms * mod[0] * mod[1] * mod[2], 2], dtype="<U5"
        )
        for i in range(mod[0]):
            for j in range(mod[1]):
                for k in range(mod[2]):
                    tmp_atom_coords[
                        self.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k) : self.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k + 1)
                    ] = self.atom_coords + (
                        self.cell[0, :] * i
                        + self.cell[1, :] * j
                        + self.cell[2, :] * k
                    )
                    tmp_atom_species[
                        self.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k) : self.Natoms
                        * (mod[2]*mod[1]*i + mod[2]*j + k + 1)
                    ] = self.atom_species
        expanded_structure = Structure.from_data(tmp_cell, tmp_atom_coords, tmp_atom_species)
        return expanded_structure
    
    def get_spglib_cell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function returns a tuple with the needed information/format for Spglib.

        Returns
        -------
            cell : np.ndarray. [3,3].
                Unit cell of the Structure object in Angstroms.
            frac_coords : np.ndarray. [Natoms,3].
                Atomic coordinates in fractional units of the Structure object.
            atom_ids : np.ndarray. [Natoms].
                Atom id of each of the atoms in the unit cell.
        """
        return (self.cell, _cell.cart2cryst(self.atom_coords, self.cell), self.atom_species[:,1])
    
    def _fix_coords(self) -> None:
        """
        This function writes the atomic coordinates of the system in a systematic manner. In converts
        the coordinates to fractional units, and then translate to the first unit-cell with x inside [0,1).
        """
        for atom in range(self.Natoms):
            coords_frac = _cell.cart2cryst(self.atom_coords[atom,:], self.cell)
            self.atom_coords[atom,:] = _cell.cryst2cart(_classify._to_unit_cell(coords_frac), self.cell)

    def Symmetrize(self, symprec: float = 1e-5) -> None:
        """
        This function symmetrizes the Structure to the symmetries specified by the symprec
        parameter. It uses Spglib in the symmetry detection.

        Parameters
        ----------
            symprec: float. Defaults to 1e-5.
                Precision in the detection of symmetries.
        """
        syms = spglib.get_symmetry(self.get_spglib_cell(), symprec=symprec)
        R, T = syms["rotations"], syms["translations"]

        ratom = np.empty([len(syms["rotations"]), self.Natoms, 3])
        for ri in range(len(R)):
            for ai, atom in enumerate(self.atom_coords):
                ratom[ri, ai] = _cell.cryst2cart(
                    (np.matmul(R[ri], _cell.cart2cryst(atom, self.cell))).transpose()
                    + T[ri],
                    self.cell,
                )

        total_rel_pos = np.zeros(self.atom_coords.shape)

        for ri in range(len(R)):
            tmp_struct = Structure.from_data(self.cell, ratom[ri], self.atom_species)
            id_list, rel_pos = _cell.map2structure(tmp_struct, self, symprec)
            total_rel_pos -= rel_pos
        self.atom_coords += total_rel_pos / len(R)

    def to_primitive(self, symprec: float = 1e-5, rotate : bool = True) -> "Structure":
        """
        This function converts the Structure object into a primitve cell detected by Spglib
        with the precision given by symprec. The cell is by default rotated to the definitions
        of the primitive cell. For comparing structures/brillouin zone, rotation can be disabled.

        Parameters
        ----------
            symprec: float. Defaults to 1e-5.
                Precision in the detection of symmetries.
            rotate: bool. Defaults to True.
                Is the cell is rotated according to the proper defintion of the primitive unit-cell.
        Returns
        -------
            Structure: mnx.Structure object.
                Structure object with the cell adjusted.
        """
        if rotate:
            lattice, frac_coords, ids = spglib.find_primitive(
                (self.cell, _cell.cart2cryst(self.atom_coords, self.cell), self.atom_species[:,1]), symprec=symprec
            )
        else:
            lattice, frac_coords, ids = spglib.standardize_cell((self.cell, _cell.cart2cryst(self.atom_coords, self.cell), self.atom_species[:,1]), symprec=symprec,to_primitive=1, no_idealize=1)

        Natoms = frac_coords.shape[0]
        atom_species = np.empty([Natoms,2], dtype="<U5")
        for atom in range(Natoms):
            atom_species[atom,0] = next((k for k, v in self._atomic_dict.items() if v == ids[atom]), None)
            atom_species[atom,1] = ids[atom]
        tmp_structure = Structure.from_data(lattice, _cell.cryst2cart(frac_coords, lattice), atom_species)
        return tmp_structure

    def change_cell(self, cell : np.ndarray) -> None:
        """
        This function changes the cell manually, moving the atomic coordinates by fixing the fractional coordinates.

        Parameters
        ----------
            cell : np.ndarray. [3,3].
                New unit-cell.
        """
        self.atom_coords = _cell.cryst2cart(_cell.cart2cryst(self.atom_coords,self.cell),cell)
        self.cell = cell
        self.rcell = _cell.get_rcell(self.cell)

    def copy(self) -> "Structure":
        """
        This function returns a copy of the Structure object.

        Returns
        -------
            Structure: mnx.Structure object.
                A copy of the Structure object.
        """
        return copy.deepcopy(self)
    
    def plot_bz(self, ax: object = None, color: str = "black", kpoints: list = None, labels: list = None) -> object:
        """
        This function visualizes the Brillouin zone of a given structure.
        
        Paramters
        ---------
            ax: matploblib axis. Defaults to None.
                The axis in which the BZ is plotted.
            color: str. Default to "black".
                Color for the BZ illustration.
            kpoints: list of floats. [Nkpoint,3]. Defaults to None.
                kpoints in fractional units to be included.
            labels: list of str. [Nkpoint]. Defaults to None.
                labels for the kpoints.
        Retruns
        -------
            ax: matplotlib axis.
                Returns a matplotlib axis with the BZ included.
        """

        from scipy.spatial import Voronoi
        import matplotlib.pyplot as plt

        qpoints = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    qpoints.append((i-2)*self.rcell[0, :]+(j-2)*self.rcell[1, :]+(k-2)*self.rcell[2,:])
        bz = Voronoi(qpoints)

        firstbz = []
        dist = 9999999
        for reg in bz.regions:
            if not reg or -1 in reg:
                continue
            nd = 0
            c = 0
            for tupl in bz.ridge_vertices:
                for i in range(len(tupl)):
                    # THE FIX: Use modulo % to connect the last vertex back to the first
                    pair = np.array([tupl[i], tupl[(i + 1) % len(tupl)]])
                    if pair[0] in reg and pair[1] in reg:
                        nd += np.linalg.norm(bz.vertices[pair[0]])
                        nd += np.linalg.norm(bz.vertices[pair[1]])
                        c += 2
            if c == 0:
                continue
            if nd / c < dist:
                dist = nd / c
                firstbz = reg

        data = []
        for tupl in bz.ridge_vertices:
            if not tupl:
                continue
            # THE FIX: Use modulo % here as well to ensure all ridge lines are closed
            for i in range(len(tupl)):
                pair = np.array([tupl[i], tupl[(i + 1) % len(tupl)]])
                if pair[0] in firstbz and pair[1] in firstbz:
                    data.append([bz.vertices[pair[0]][0], bz.vertices[pair[0]][1], bz.vertices[pair[0]][2]])
                    data.append([bz.vertices[pair[1]][0], bz.vertices[pair[1]][1], bz.vertices[pair[1]][2]])
        
        data = np.asarray(data)
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_proj_type("ortho")
            for a, b in zip(data[::2], data[1::2]):
                c = np.vstack([a, b])
                ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color)
            
            # Use data max for better scaling
            s = np.max(np.abs(data))
            ax.set_xlim(-1.2*s,1.2*s), ax.set_ylim(-1.2*s,1.2*s), ax.set_zlim(-1.2*s,1.2*s)
            ax.set_axis_off()
        else:
            for a, b in zip(data[::2], data[1::2]):
                c = np.vstack([a, b])
                ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color)
        
        try:
            if kpoints is not None:
                kpoints, labels = np.array(kpoints), np.array(labels)
                for qi, q in enumerate(kpoints):
                    # Manual cartesian conversion to ensure it works without external util
                    q_cart = np.dot(q, self.rcell)
                    ax.scatter(q_cart[0], q_cart[1], q_cart[2], color=color)
                    if labels is not None:
                        ax.text(q_cart[0], q_cart[1], q_cart[2]+0.001, s=labels[qi], color=color)
        except:
            pass
            
        return(ax)


def _get_atom_species_from_atom_numbers(atom_numbers):
    atom_species = np.empty([len(atom_numbers), 2], dtype="<U5")
    keys = list(ase.data.atomic_numbers.keys())
    values = list(ase.data.atomic_numbers.values())
    atomic_species_dict = {}
    id = 0
    for atom in range(len(atom_numbers)):
        atom_species[atom, 0] = keys[values.index(atom_numbers[atom])]
        if atom_species[atom, 0] not in list(atomic_species_dict.keys()):
            id += 1
            atomic_species_dict[str(atom_species[atom, 0])] = id
        atom_species[atom, 1] = atomic_species_dict[atom_species[atom, 0]]
    return atom_species

