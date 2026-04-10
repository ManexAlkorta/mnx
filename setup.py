import setuptools
from numpy.distutils.core import setup, Extension
from numpy.distutils.system_info import get_info

with open("README.md", "r") as fh:
    long_description = fh.read()

lapack_opt = get_info('lapack_opt')

# Define the Fortran Extension
# name: 'mnx.phonon_lib' means it will be inside the mnx package
ext1 = Extension(
    name='mnx.FModules',
    sources=['src/mnx/FModules/interpolation.f90'],
    extra_f90_compile_args=["-O3"], # Optional: optimization
    **lapack_opt
)

ext2 = Extension(
    name="symph",
    sources=[
        "src/mnx/FModules/QE/symdynph_gq_new.f90", "src/mnx/FModules/QE/symm_base.f90", 
        "src/mnx/FModules/QE/sgam_ph.f90", "src/mnx/FModules/QE/invmat.f90", "src/mnx/FModules/QE/set_asr.f90",
        "src/mnx/FModules/QE/error_handler.f90", "src/mnx/FModules/QE/io_global.f90",
        "src/mnx/FModules/QE/flush_unit.f90", "src/mnx/FModules/QE/symvector.f90",
        "src/mnx/FModules/QE/fc_supercell_from_dyn.f90", "src/mnx/FModules/QE/set_tau.f90",
        "src/mnx/FModules/QE/cryst_to_car.f90", "src/mnx/FModules/QE/recips.f90",
        "src/mnx/FModules/QE/q2qstar_out.f90", "src/mnx/FModules/QE/rotate_and_add_dyn.f90",
        "src/mnx/FModules/QE/trntnsc.f90", "src/mnx/FModules/QE/star_q.f90", "src/mnx/FModules/QE/eqvect.f90",
        "src/mnx/FModules/QE/symm_matrix.f90", "src/mnx/FModules/QE/from_matdyn.f90",
        "src/mnx/FModules/QE/interp.f90", "src/mnx/FModules/QE/q_gen.f90", "src/mnx/FModules/QE/smallgq.f90",
        "src/mnx/FModules/QE/symmetry_high_rank.f90", "src/mnx/FModules/QE/unwrap_tensors.f90",
        "src/mnx/FModules/QE/get_latvec.f90", "src/mnx/FModules/QE/contract_two_phonon_propagator.f90",
        "src/mnx/FModules/QE/get_q_grid_fast.f90", "src/mnx/FModules/QE/kind.f90",
        "src/mnx/FModules/QE/constants.f90", "src/mnx/FModules/QE/eff_charge_interp.f90",
        "src/mnx/FModules/QE/get_translations.f90", "src/mnx/FModules/QE/get_equivalent_atoms.f90"
    ],    extra_f90_compile_args=["-cpp"],
    **lapack_opt
)

setup(
    name="mnx",
    version="0.1.0",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    # This is the critical line to include the compiled Fortran
    ext_modules=[ext1,ext2],
    author="Manex Alkorta Lopetegi",
    author_email="manexalk@gmail.com",
    description="Phonon tools with Fortran backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "spglib",
        "ase",
        "PyQt5", #Interactive plots.
    ],
    python_requires='>=3.6',
)
