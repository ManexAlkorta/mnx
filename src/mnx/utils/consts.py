##################################################
## This file contains the constants used in mnx ##
##################################################

bohr2angstroms = 0.52917721067121
angstroms2bohr = 1.8897259886
Ry2eV = 13.6056980659
eV2Ry = 0.0734985857
eV2cm = 8065.544
Ry2cm = Ry2eV*eV2cm
bohr = 5.2917721067121e-11
cm2Thz = 0.0299792458

Ry2jul= 2.179872e-18
jul2eV = Ry2eV/Ry2jul
ang2metre=1e-10

eVA2GPa = 16.021766208

# Force
eVA2RyBohr = 0.038893793468002

# Force derivatives
eVA2RyBohr2 =  eVA2RyBohr * bohr2angstroms

# Mass I think this is wrong jeje
e_mass = 1822.8885468045 # in atomic mass units
Ry2AMU = 2 / e_mass
