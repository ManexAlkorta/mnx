import numpy as np
import sisl as sisl
import ase.data
import ase.io
import spglib
from ase import Atoms

import mnx.utils.cell_utils as cu


class STRUCT:
    """STRUCTURE OBJECT. The STRUCT class contains information about the 
    atomic species and atomic coordinates.

    Main attributes:
        self.atom_coords: Atomic coordinates in Angstromns. np.array.
            [Natoms, 3].
        self.atom_species: It contains information about the atom type.
            np.array. [Natoms, 3]. It is structured as follows:
                [:, 0]: Element name, i.e. "Cs".
                [:, 1]: Atom id, i.e. "1".
                [:, 2]: Atomic number taken from ASE.
        
    """

    def __init__(self, file="", format="", ASE=False, ASE_struct=None, dyn=None):

        if ASE:
            self.cell_units = "angstrom"
            self.cell = np.array(ASE_struct.get_cell())
            self.atom_coords = np.array(ASE_struct.get_positions())  # A
            try:
                self.atom_forces = np.array(ASE_struct.get_forces())  # eV/A
                tmp_stress = -ASE_struct.get_stress()  # ASE sign convention. eV/A^3
                self.stress = np.array(
                    [
                        [tmp_stress[0], tmp_stress[5], tmp_stress[4]],
                        [tmp_stress[5], tmp_stress[1], tmp_stress[3]],
                        [tmp_stress[4], tmp_stress[3], tmp_stress[2]],
                    ]
                )
                self.Etot = ASE_struct.get_potential_energy()  # eV
            except:
                None
            self.Natoms = ASE_struct.get_global_number_of_atoms()
            self.atom_species = np.empty([self.Natoms, 3], dtype="<U5")
            atomic_species_dict, k = {}, 1
            for atom in range(self.Natoms):
                if ASE_struct.symbols[atom] not in self.atom_species[:, 0]:
                    atomic_species_dict[ASE_struct.symbols[atom]] = k
                    k += 1
                self.atom_species[atom, 0] = ASE_struct.symbols[atom]
                self.atom_species[atom, 1] = atomic_species_dict[
                    ASE_struct.symbols[atom]
                ]
                self.atom_species[atom, 2] = ASE_struct.get_atomic_numbers()[atom]
            list_species = []
            for atom in range(self.Natoms):
                if self.atom_species[atom, 0] not in list_species:
                    list_species.append(self.atom_species[atom, 0])
            self.Nspecies = len(list_species)
            self.rcell = cu.get_rcell(self.cell)
        else:
            if format == "qe":
                ###############################################################
                ### The dynamic matrix is asked to read the atomic species ####
                ###############################################################
                data = (open(file, "r")).readlines()
                atomic_species_dict = {}
                self.cell = np.empty([3, 3], dtype=float)
                i = 0
                while data[i][:15] != "CELL_PARAMETERS":
                    i += 1
                self.cell_units = data[i].split()[-1]
                i += 1
                for j in range(3):
                    for k in range(3):
                        self.cell[j, k] = float((data[i + j].split())[k])
                i += 3
                self.rcell = cu.get_rcell(self.cell)
                while data[i][:16] != "ATOMIC_POSITIONS":
                    i += 1
                i += 1
                self.Natoms = len(data) - i  # IDK
                self.atom_species = np.empty([self.Natoms, 3], dtype="<U5")
                self.atom_coords = np.empty([self.Natoms, 3], dtype=float)
                tmp_id = 0
                for j in range(i, self.Natoms + i):
                    atom = j - i
                    self.atom_species[atom, 0] = (data[j].split())[0]
                    if self.atom_species[atom, 0] not in list(
                        atomic_species_dict.keys()
                    ):
                        atomic_species_dict[self.atom_species[atom, 0]] = tmp_id
                        tmp_id += 1
                    self.atom_species[atom, 1] = atomic_species_dict[
                        self.atom_species[atom, 0]
                    ]
                    self.atom_species[atom, 2] = ase.data.atomic_numbers[
                        str(self.atom_species[atom, 0])
                    ]
                    for k in range(3):
                        self.atom_coords[atom, k] = (data[j].split())[k + 1]
                self.Nspecies = tmp_id

            elif format == "siesta":
                data = (open(file, "r")).readlines()
                i = 0
                while data[i][:-1].split()[0] != "LatticeConstant":
                    i += 1
                if (
                    data[i][:-1].split()[-1] == "Ang"
                    and float(data[i][:-1].split()[1]) == 1
                ):
                    self.cell_units = "angstrom"
                else:
                    print("Not implemented cell definition.")
                    raise ValueError
                self.cell = np.empty([3, 3], dtype=float)
                while data[i][:-1] != "%block LatticeVectors":
                    i += 1
                i += 1
                for j in range(3):
                    for k in range(3):
                        self.cell[j, k] = float((data[i + j].split())[k])
                i += 3
                self.rcell = cu.get_rcell(self.cell)
                while data[i][0:13] != "NumberOfAtoms":
                    i += 1
                self.Natoms = int(data[i].split()[-1])
                while data[i][:-1].split()[0] != "AtomicCoordinatesFormat":
                    i += 1
                coords_units = data[i][:-1].split()[1]
                self.atom_coords = np.empty([self.Natoms, 3])
                self.atom_species = np.empty([self.Natoms, 3], dtype="<U5")
                while data[i][:-1] != "%block AtomicCoordinatesAndAtomicSpecies":
                    i += 1
                i += 1
                for atom in range(self.Natoms):
                    self.atom_coords[atom][:] = np.array(
                        data[i + atom].split()[:3], dtype=float
                    )
                    self.atom_species[atom][0] = data[i + atom].split()[6]
                    self.atom_species[atom][1] = data[i + atom].split()[3]
                    try:
                        self.atom_species[atom][2] = ase.data.atomic_numbers[
                            str(self.atom_species[atom][0])
                        ]
                    except KeyError:
                        print("WARNING: Atomic label not recognized by ASE. Set to H.")
                        self.atom_species[atom][2] = 1
                if coords_units == "Ang":
                    None
                elif coords_units == "Fractional":
                    self.atom_coords = cu.cryst2cart(self.atom_coords, self.cell)
                while data[i][:15] != "NumberOfSpecies":
                    i += 1
                self.Nspecies = int((data[i][:-1]).split()[-1])
            elif format == "vasp":
                tmp_ase_struct = ase.io.read(file)
                tmp_struct = STRUCT(ASE=True, ASE_struct=tmp_ase_struct)
                self.atom_coords = tmp_struct.atom_coords
                self.cell = tmp_struct.cell
                self.Natoms = tmp_struct.Natoms
                self.Nspecies = tmp_struct.Nspecies
                self.atom_species = tmp_struct.atom_species


    def set_cell(self, cell, units):
        """Set the attribute cell of the STRUCT object.
        
        Input:
            cell: The unit cell of the structure. np.array. [3, 3].
            units: The units in which the cell is given. str.
        Returns:
            None
        """
        self.cell = cell
        self.cell_units = units

    def set_atom_coords(self, atom_coords, units="angstrom"):
        """Set the atomic coordinates of the STRUCT object.
        
        Input:
            atom_coords: The atomic coordinates. np.array. [Natoms, 3].
            units: The units in which the coordinates are given. By now angstrom
                or fractional.
        
        Returns:
            None
        """
        self.Natoms = len(atom_coords[:, 0])
        if units == "angstrom":
            self.atom_coords = np.array(atom_coords, dtype=float)
        elif units == "fractional":
            self.atom_coords = np.empty([self.Natoms, 3])
            for atom in range(self.Natoms):
                self.atom_coords[atom] = np.matmul(
                    self.cell.transpose(), np.array(atom_coords[atom], dtype=float)
                )
        else:
            raise NotImplemented

    def set_atom_forces(self, forces, units="eV/angstrom"):
        """Set the atomic forces of the STRUCT object.
        
        Input:
            forces: The atomic forces. np.array. [Natoms, 3].
            units: The units in which the atomic forces are given. str.
        
        Returns:
            None
        """
        if units == "eV/angstrom":
            self.atom_forces = forces
        else:
            raise NotImplemented

    def set_stress(self, stress, units="eV/angstrom^3"):
        """Set the stress tensor of the STRUCT object.
        
        Input:
            stress: The stress tensor. np.array. [3,3].
            units: The units in which the stress tensor is given.
        
        Returns:
            None
        """
        if units == "eV/angstrom^3":
            self.stress = stress
        elif units == "eV/u.c.":
            self.stress = stress / self.vol
        else:
            raise NotImplemented

    def set_vol(self, vol, units="angstrom^3"):
        """Set the volume of the unit cell of the STRUCT object.
        
        Input:
            vol: Volume of the unit cell. float.
            units: Units in which the volume is given. str.

        Returns:
            None
        """
        if units == "angstrom^3":
            self.vol = vol
        else:
            raise NotImplemented

    def set_Etot(self, Etot, units="eV"):
        """Set the total energy that corresponds to the STRUCT object.
        
        Input:
            Etot: Total energy. float.
            units: Units in which the total energy is given. str.
        
        Returns:
            None
        """
        if units == "eV":
            self.Etot = Etot
        else:
            raise NotImplemented

    def set_Nspecies(self, Nspecies):
        """Set the number of species of the STRUCT object.
        
        Input:
            Nspecies: The total number of different species. int.
        
        Returns:
            None
        """
        self.Nspecies = Nspecies

    def set_atom_species(self, atom_species):
        """Set the atom_species array of the STRUCT object.
        
        Input:
            atom_species: It contains information about the atom type.
                np.array. [Natoms, 3]. It is structured as follows:
                    [:, 0]: Element name, i.e. "Cs".
                    [:, 1]: Atom id, i.e. "1".
                    [:, 2]: Atomic number taken from ASE.
        
        Returns:
            None
        """
        self.atom_species = atom_species
        self.Natoms = len(atom_species[:, 0])
        id_list = []
        for atom in range(self.Natoms):
            if self.atom_species[atom, 1] not in id_list:
                id_list.append(self.atom_species[atom, 1])
        self.Nspecies = len(id_list)

    def set_atom_species_from_atom_numbers(self, atom_numbers):
        """Set the atom species of the STRUCT object from the atomic number.
        
        Input:
            atom_number: atomic number of each atom in the cell. np.array. [Natoms].
        
        Returns:
            None
        """
        self.atom_species = np.empty([len(atom_numbers), 3], dtype="<U5")
        keys = list(ase.data.atomic_numbers.keys())
        values = list(ase.data.atomic_numbers.values())
        atomic_species_dict = {}
        id = 0
        for atom in range(len(atom_numbers)):
            self.atom_species[atom, 0] = keys[values.index(atom_numbers[atom])]
            if self.atom_species[atom, 0] not in list(atomic_species_dict.keys()):
                id += 1
                atomic_species_dict[str(self.atom_species[atom, 0])] = id
            self.atom_species[atom, 1] = atomic_species_dict[self.atom_species[atom, 0]]
            self.atom_species[atom, 2] = atom_numbers[atom]
        self.Nspecies = id

    def set_Natoms(self, Natoms):
        self.Natoms = Natoms


    def write(self, file, format=""):
        """This function writes the STRUCT object in a formatted text file.
        
        Input:
            file: Name of the outfile. str.
            format: Format in which to write the outfile. str.
        
        Return:
            None
        """
        if format == "qe":
            fmt_cell = "{0:18.9f}{1:18.9f}{2:18.9f}\n"
            fmt_atom_struct = "{0:>10}{1:18.9f}{2:18.9f}{3:18.9f}\n"
            file = open(file, "w")
            file.write("ATOMIC_POSITIONS {crystal}\n")
            atom_coords_crystal = cu.cart2cryst(self.atom_coords, self.cell)
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
        elif format == "siesta":
            fmt_chemical_species = "{0:>4d}{1:>4d}{2:>4}\n"
            fmt_cell = "{0:18.9f}{1:18.9f}{2:18.9f}\n"
            fmt_atom_struct = "{0:18.9f}{1:18.9f}{2:18.9f} {3:>4d} # {4:>4d} {5:>3}\n"
            file = open(file, "w")
            if self.cell_units != "angstrom":
                print("Only cells in Angstroms are implemented!")
                raise TypeError
            file.write("LatticeConstant 1.0 Ang\n")
            file.write("%block LatticeVectors\n")
            for i in range(3):
                file.write(
                    fmt_cell.format(self.cell[i, 0], self.cell[i, 1], self.cell[i, 2])
                )
            file.write("%endblock LatticeVectors\n\n")
            file.write("NumberOfAtoms {}\n".format(self.Natoms))
            file.write("AtomicCoordinatesFormat Ang\n")
            file.write("%block AtomicCoordinatesAndAtomicSpecies\n")
            for atom in range(self.Natoms):
                file.write(
                    fmt_atom_struct.format(
                        self.atom_coords[atom, 0],
                        self.atom_coords[atom, 1],
                        self.atom_coords[atom, 2],
                        int(self.atom_species[atom, 1]),
                        atom,
                        self.atom_species[atom, 0],
                    )
                )
            file.write("%endblock AtomicCoordinatesAndAtomicSpecies\n\n")
            file.write("NumberOfSpecies {}\n".format(self.Nspecies))
            file.write("%block ChemicalSpeciesLabel\n")
            list_species = []
            for atom in range(self.Natoms):
                if self.atom_species[atom, 0] not in list_species:
                    file.write(
                        fmt_chemical_species.format(
                            int(self.atom_species[atom, 1]),
                            int(self.atom_species[atom, 2]),
                            self.atom_species[atom, 0],
                        )
                    )
                    list_species.append(self.atom_species[atom, 0])
            file.write("%endblock ChemicalSpeciesLabel\n")
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

    def toASE(self):
        """Returns an ASE atom object from an STRUCT object.
        
        Input:
            None
        
        Returns:
            ASE_struct: ASE atom object.
        """
        symbols = ""
        for symbol in self.atom_species[:, 0]:
            symbols += symbol
        ASE_struct = Atoms(
            symbols=symbols,
            positions=self.atom_coords,
            cell=self.cell,
            pbc=(True, True, True),
        )
        return ASE_struct

    def get_spacegroup(self, symprec=1e-5):
        """Spglib function get_spacegroup.

        Input:
            symprec: Symmetry precision for spglib. float.
        
        Return:
            space_group: The detected spacegroup by spglib. str.
        """
        return spglib.get_spacegroup(self.toASE(), symprec=symprec)

    def to_primitive(self, symprec=1e-5):
        """This fuction changes the cell of the STRUCT object
        into the primitive cell given by spglib.
        
        Input:
            symprec: Symmetry precision for spglib. float.
        
        Return:
            None
        """
        lattice, frac_coords, numbers = spglib.find_primitive(
            self.toASE(), symprec=symprec
        )
        self.set_cell(cell=lattice, units="angstrom")
        self.set_atom_coords(frac_coords, units="fractional")
        self.set_atom_species_from_atom_numbers(numbers)

    def symmetrize(self, symprec=1e-5, loops=1):
        """Symmetrize the structure to the symmetries found by spglib.

        Input:
            symprec: Symmetry precision for spglib. float.
            loops: The number of symmetrization iterations.
        
        Returns:
            None
        """
        syms = spglib.get_symmetry(self.toASE(), symprec=symprec)
        R, T = syms["rotations"], syms["translations"]
        for loop in range(loops):
            atom_coords = np.zeros([self.Natoms, 3])
            ratom = np.empty([len(syms["rotations"]), self.Natoms, 3])
            for ri in range(len(R)):
                for ai, atom in enumerate(self.atom_coords):
                    ratom[ri, ai] = cu.cryst2cart(
                        (np.matmul(R[ri], cu.cart2cryst(atom, self.cell))).transpose()
                        + T[ri],
                        self.cell,
                    )
            for ri in range(len(R)):
                tmp_struct = STRUCT()
                tmp_struct.set_cell(self.cell, units="angstrom")
                tmp_struct.set_atom_coords(ratom[ri])
                tmp_struct.set_atom_species(self.atom_species)
                tmp_struct.set_atom_forces(np.zeros([self.Natoms, 3]))
                id_list, translations = cu.map2structure(self, tmp_struct)
                tmp_struct.reorder2list(id_list, translations)
                atom_coords += tmp_struct.atom_coords / len(R)
            self.set_atom_coords(atom_coords, units="angstrom")

    # def reorder2list(self, id_list, translations):
    #     """Reorder the atomic id from a list. The list and translations
    #     are obtained with the function map2structure in the utils/cell_utils.py.
        
    #     Input:
    #         id_list: List of atomic id. list. [Natoms].
    #         translations: Translations corresponding to the atom id. [Natoms, 3].
        
    #     Returns:
    #         None
    #     """
    #     tmp_atom_coords = np.empty([self.Natoms, 3])
    #     tmp_atom_forces = np.empty([self.Natoms, 3])
    #     tmp_atom_species = np.empty([self.Natoms, 3], dtype="<U5")
    #     for i in range(self.Natoms):
    #         tmp_atom_coords[i] = self.atom_coords[id_list[i]] - cu.cryst2cart(
    #             translations[i], self.cell
    #         )
    #         tmp_atom_forces[i] = self.atom_forces[id_list[i]]
    #         tmp_atom_species[i] = self.atom_species[id_list[i]]
    #     self.set_atom_coords(tmp_atom_coords)
    #     self.set_atom_species(tmp_atom_species)
    #     self.set_atom_forces(tmp_atom_forces)

    # def get_rcell(self, alat=False):
    #     ktea = 1
    #     rcell = np.empty([3, 3])
    #     for k in range(3):
    #         s = np.zeros([3])
    #         s[k] = ktea
    #         if alat:
    #             rcell[k, :] = np.matmul(
    #                 np.linalg.inv(self.cell / np.linalg.norm(self.cell[0])), s
    #             )
    #         else:
    #             rcell[k, :] = np.matmul(np.linalg.inv(self.cell), s)
    #     return rcell

    # def cart2cryst(self, xyz=None, reciprocal=False, alat=False, cell=None):
    #     try:
    #         if cell == None:
    #             cell_bool = False
    #         else:
    #             cell_bool = True
    #     except ValueError:
    #         cell_bool = True
    #     try:
    #         if xyz == None:
    #             xyz_bool = False
    #         else:
    #             xyz_bool = True
    #     except ValueError:
    #         xyz_bool = True
    #     if not cell_bool:
    #         if not xyz_bool:
    #             coords_crystal = np.empty([self.Natoms, 3])
    #             M = np.linalg.inv(self.cell)
    #             for atom in range(self.Natoms):
    #                 coords_crystal[atom] = np.matmul(
    #                     M.transpose(), self.atom_coords[atom]
    #                 )
    #         else:
    #             if reciprocal:
    #                 if alat:
    #                     coords_crystal = np.empty([3])
    #                     M = np.linalg.inv(cu.get_rcell(self.cell, alat=alat))
    #                     coords_crystal = np.matmul(M.transpose(), xyz)
    #                 else:
    #                     coords_crystal = np.empty([3])
    #                     M = np.linalg.inv(self.rcell)
    #                     coords_crystal = np.matmul(M.transpose(), xyz)
    #             else:
    #                 coords_crystal = np.empty([3])
    #                 M = np.linalg.inv(self.cell)
    #                 coords_crystal = np.matmul(M.transpose(), xyz)
    #     else:
    #         coords_crystal = np.empty([3])
    #         M = np.linalg.inv(cell.transpose())
    #         coords_crystal = np.matmul(M, xyz.transpose()).transpose()
    #     return coords_crystal

    # def cryst2cart(self, xyz=None, reciprocal=False, alat=False, cell=None):
    #     try:
    #         if cell == None:
    #             cell_bool = False
    #         else:
    #             cell_bool = True
    #     except ValueError:
    #         cell_bool = True
    #     try:
    #         if xyz == None:
    #             xyz_bool = False
    #         else:
    #             xyz_bool = True
    #     except ValueError:
    #         xyz_bool = True
    #     if not cell_bool:
    #         if not xyz_bool:
    #             coords_crystal = np.empty([self.Natoms, 3])
    #             M = self.cell
    #             for atom in range(self.Natoms):
    #                 coords_crystal[atom] = np.matmul(
    #                     M.transpose(), self.atom_coords[atom]
    #                 )
    #         else:
    #             if reciprocal:
    #                 if alat:
    #                     coords_crystal = np.empty([3])
    #                     M = cu.get_rcell(self.cell, alat=alat)
    #                     coords_crystal = np.matmul(M.transpose(), xyz)
    #                 else:
    #                     coords_crystal = np.empty([3])
    #                     M = self.rcell
    #                     coords_crystal = np.matmul(M.transpose(), xyz)
    #             else:
    #                 coords_crystal = np.empty([3])
    #                 M = self.cell
    #                 coords_crystal = np.matmul(M.transpose(), xyz)
    #     else:
    #         coords_crystal = np.empty([3])
    #         M = cell
    #         coords_crystal = np.matmul(M.transpose(), xyz)

    #     return coords_crystal