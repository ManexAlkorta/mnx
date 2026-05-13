project(
  'mnx',
  'c',
  'fortran',
  version : '0.1.0',
  default_options : ['buildtype=release']
)

# =============================================================================
# 1. Environment & Python Setup
# =============================================================================
py_mod = import('python')
py = py_mod.find_installation(pure : false)
py_dep = py.dependency() 

inc_np = run_command(py, '-c', 'import numpy; print(numpy.get_include())', check : true).stdout().strip()
inc_f2py = run_command(py, '-c', 'import numpy.f2py; print(numpy.f2py.get_include())', check : true).stdout().strip()

fortranobject_c = inc_f2py + '/fortranobject.c'
numpy_includes = include_directories(inc_np, inc_f2py)

lapack_dep = dependency('lapack', required : true)
blas_dep = dependency('blas', required : true)

install_subdir('src/mnx', install_dir : py.get_install_dir())

# =============================================================================
# 2. Extension Module: FModules
# =============================================================================
fmod_src = 'src/mnx/FModules/interpolation.f90'
fmod_gen = custom_target(
  'fmodules_f2py',
  input : fmod_src,
  output : ['FModulesmodule.c', 'FModules-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'FModules', '--lower', '--build-dir', '.'],
  build_by_default : true
)

py.extension_module(
  'FModules',
  [fmod_src, fmod_gen, fortranobject_c],
  dependencies : [py_dep, lapack_dep, blas_dep],
  include_directories : numpy_includes,
  install : true,
  subdir : 'mnx'
)

# =============================================================================
# 3. Extension Module: symph
# =============================================================================
qe_path = 'src/mnx/FModules/QE/'
symph_filenames = [
  'symdynph_gq_new.f90', 'symm_base.f90', 'sgam_ph.f90', 'invmat.f90',
  'set_asr.f90', 'error_handler.f90', 'io_global.f90', 'flush_unit.f90',
  'symvector.f90', 'fc_supercell_from_dyn.f90', 'set_tau.f90',
  'cryst_to_car.f90', 'recips.f90', 'q2qstar_out.f90', 'rotate_and_add_dyn.f90',
  'trntnsc.f90', 'star_q.f90', 'eqvect.f90', 'symm_matrix.f90',
  'from_matdyn.f90', 'interp.f90', 'q_gen.f90', 'smallgq.f90',
  'symmetry_high_rank.f90', 'unwrap_tensors.f90', 'get_latvec.f90',
  'contract_two_phonon_propagator.f90', 'get_q_grid_fast.f90', 'kind.f90',
  'constants.f90', 'eff_charge_interp.f90', 'get_translations.f90',
  'get_equivalent_atoms.f90'
]

symph_sources = []
foreach f : symph_filenames
  symph_sources += qe_path + f
endforeach

# Point Meson to look for the signature file directly in the QE directory
pyf_blueprint = files(qe_path + 'symph.pyf')

symph_gen = custom_target(
  'symph_f2py',
  input : pyf_blueprint,
  output : ['symphmodule.c', 'symph-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '--lower', '--build-dir', '.'],
  build_by_default : true
)

py.extension_module(
  'symph',
  [symph_sources, symph_gen, fortranobject_c],
  dependencies : [py_dep, lapack_dep, blas_dep],
  include_directories : numpy_includes,
  fortran_args : ['-cpp'],
  install : true,
  subdir : 'mnx'
)