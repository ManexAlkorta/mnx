import numpy as np
import ase.io
import os

from mnx.struct_handlers.dyn_class import DYN_MATRIX
from mnx.struct_handlers.struct_class import STRUCT
import mnx.utils.cell_utils as cu
import mnx.utils.consts as consts


class POPULATION:
    """POPULATION object. The POPULATION object contains the info of an already
    calculated SSCHA population.
    
    Main attributes:
        self.dyn0, self.struct0: The referenct DYN_MATRIX and STRUCT objects.
        self.xats: Distortion from the reference structure. np.array. [N, Natoms, 3].
        self.energies: Energy of each configuration of the population. np.array. [N].
        self.forces: Forces of each configuration of the population. np.array. [N, Natoms, 3].
        self.stress: Stress tensor of each configuration of the population. np.array. [3, 3].
    """


    def __init__(self, pop_id, file, mod=[1, 1, 1], format="xyz"):
        self.dyn0 = DYN_MATRIX(f"./dyn_start_population{pop_id}_1")
        self.struct0 = self.dyn0.distort_structure([0], [[0, 0]], mod=mod)
        self.pop_id = pop_id
        if format == "xyz":
            self.atoms = ase.io.read(file, ":")
            self.N = len(self.atoms)
            self.Natoms = len(self.atoms[0].symbols)
            self.energies = np.empty([self.N])
            self.forces = np.empty([self.N, self.Natoms, 3])
            self.xats = np.empty([self.N, self.Natoms, 3])
            self.du = np.empty([self.N, self.Natoms, 3])
            self.stress = np.empty([self.N, 3, 3])
            self.struct_list = []
            tmp_struct = STRUCT(ASE=True, ASE_struct=self.atoms[0])
            id_list, translations = cu.map2structure(self.struct0, tmp_struct)
            for i, conf in enumerate(self.atoms):
                # Save the population in SSCHA units
                tmp_struct = STRUCT(ASE=True, ASE_struct=conf)
                tmp_struct.reorder2list(id_list, translations)
                self.struct_list.append(tmp_struct)
                self.energies[i] = tmp_struct.Etot  # eV
                self.forces[i] = tmp_struct.atom_forces  # eV/A
                self.du[i] = cu.corrected_displacements(tmp_struct, self.struct0)  # A
                self.xats[i] = (
                    self.struct0.atom_coords + self.du[i]
                )  # To make it readable for CC.
                self.stress[i] = tmp_struct.stress  # eV/A^3

    def save_bin(self, folder="./data_ensemble/", N=0):
        """This fucntion saves the population data in numpy
        binary files.
        
        Input:
            folder: Folder in which the data will be saved. str.
            N: Up to which configuration of the population to save the data.
                Usefull to test convergence the number of configuraions. int.
        
        Returns:
            None
        """

        it_exits = os.path.exists(folder)
        if not it_exits:
            os.makedirs(folder)
        if N == 0:
            np.save(
                f"{folder}/energies_pop{self.pop_id}.npy", self.energies * consts.eV2Ry
            )  # Ry
            np.save(
                f"{folder}/forces_pop{self.pop_id}.npy", self.forces * consts.eV2Ry
            )  # Ry/A
            np.save(
                f"{folder}/stresses_pop{self.pop_id}.npy",
                self.stress * consts.eV2Ry / (consts.angstroms2bohr) ** 3,
            )  # Ry/Bohr^3
            np.save(f"{folder}/xats_pop{self.pop_id}.npy", self.xats)  # A
        else:
            np.save(
                f"{folder}/energies_pop{self.pop_id}.npy",
                self.energies[:N] * consts.eV2Ry,
            )  # Ry
            np.save(
                f"{folder}/forces_pop{self.pop_id}.npy", self.forces[:N] * consts.eV2Ry
            )  # Ry/A
            np.save(
                f"{folder}/stresses_pop{self.pop_id}.npy",
                self.stress[:N] * consts.eV2Ry / (consts.angstroms2bohr) ** 3,
            )  # Ry/Bohr^3
            np.save(f"{folder}/xats_pop{self.pop_id}.npy", self.xats[:N])  # A

    def save(self, folder="./data_ensemble/"):
        """This fucntion saves the population data in classical SSCHA format.
        
        Input:
            folder: Folder in which the data will be saved. str.
        
        Returns:
            None
        """
        it_exits = os.path.exists(folder)
        if not it_exits:
            os.makedirs(folder)
        with open(
            folder + f"/energies_supercell_population{self.pop_id}.dat", "w+"
        ) as outfile:
            for energy in self.energies:
                outfile.write("{0:16.8f}\n".format(energy * consts.eV2Ry))
        for i, forces in enumerate(self.forces):
            with open(
                folder + f"/forces_population{self.pop_id}_{i+1}.dat", "w+"
            ) as outfile:
                for atom in range(self.Natoms):
                    outfile.write(
                        "{0:16.8f}  {1:16.8f}  {2:16.8f}\n".format(
                            forces[atom, 0] * consts.eV2Ry / consts.angstroms2bohr,
                            forces[atom, 1] * consts.eV2Ry / consts.angstroms2bohr,
                            forces[atom, 2] * consts.eV2Ry / consts.angstroms2bohr,
                        )
                    )
        for i, stress in enumerate(self.stress):
            with open(
                folder + f"/stresses_population{self.pop_id}_{i+1}.dat", "w+"
            ) as outfile:
                for k in range(3):
                    outfile.write(
                        "{0:16.8f}  {1:16.8f}  {2:16.8f}\n".format(
                            stress[k, 0] * consts.eV2Ry / (consts.angstroms2bohr) ** 3,
                            stress[k, 1] * consts.eV2Ry / (consts.angstroms2bohr) ** 3,
                            stress[k, 2] * consts.eV2Ry / (consts.angstroms2bohr) ** 3,
                        )
                    )
        for i, du in enumerate(self.du):
            with open(
                folder + f"/u_population{self.pop_id}_{i+1}.dat", "w+"
            ) as outfile:
                for atom in range(self.Natoms):
                    outfile.write(
                        "{0:16.8f}  {1:16.8f}  {2:16.8f}\n".format(
                            du[atom, 0] * consts.angstroms2bohr,
                            du[atom, 1] * consts.angstroms2bohr,
                            du[atom, 2] * consts.angstroms2bohr,
                        )
                    )
        for i, conf in enumerate(self.struct_list):
            self.struct_list[i].write(
                folder + f"/scf_population{self.pop_id}_{i+1}.dat", format="qe"
            )
