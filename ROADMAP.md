# mnx polishing roadmap

A staged plan to make **mnx** cleaner, faster and easier to pick up, without
changing the physics it produces. Each stage is meant to be a small, reviewable
branch. Stages are ordered so that later work rests on earlier safety nets.

Legend: ✅ done on `polish/dynmatrix` · ⬜ to do

---

## Stage 0 — `DynMatrix` / `DynQ` (`src/mnx/dyn_matrix.py`)

The phonon core, and the first module polished.

### Bugs fixed
- ✅ `DynMatrix.from_file` with `Nqirr=0` (single dyn file) crashed (`qs` undefined,
  `DynQs` not an array). It now reads one star, and only builds `phiR` when the
  `qgrid` is a real grid.
- ✅ `get_bands` / `get_bands_with_SF` crashed for some `N` (e.g. `N=30`) because the
  path was built with floating-point `np.arange`; now uses `np.linspace`.
- ✅ `DynQ.from_data` divided *every* displacement by the mass of the **last** atom.
  Displacements are now `e / sqrt(m_atom)` per atom. This changes
  `get_bands_with_SF` for systems with several species (e.g. CrSb).
- ✅ `DynQ.compute_traces` always took the non-degenerate branch, so degenerate
  modes got single diagonal elements instead of the trace over the degenerate
  block (the character). It also left unset entries uninitialised.
- ✅ `Symmetrize` rebuilt each `DynQ` with the wrong q-star (`qstars[qi]`); harmless
  only by luck. Now recomputed directly.
- ✅ `distort_structure` crashed when combining stars of different dimension
  (ragged `Q`), the case its own docstring advertises.
- ✅ Files opened with `open(...)` were never closed (`DynQ.from_file`, `DynQ.write`).
- ✅ Mutable default argument `qgrid=[0,0,0]`; `np.float128` passed to Fortran;
  `== None` comparisons; bare `except:` in `from_phonopy`.

### Efficiency
- ✅ `get_bands`, `get_bands_with_SF`: no longer build a full `DynQ` per q-point
  (which ran an spglib symmetry search every time); the Fortran interpolation is
  called directly and the negative-frequency and structure-factor code is vectorised.
- ✅ `DynQ` parsing, mass scaling, diagonalisation set-up, reordering, degeneracy
  search, polarisation-vector expansion and `get_Pmn` are vectorised with NumPy
  instead of nested Python loops.
- ✅ `DynMatrix.expand_polvecs` reuses `DynQ._expand_polvecs` instead of a copy of it.
- Measured on the tutorial data (before → after): `get_bands` CrSb 0.43 s → 0.009 s,
  `get_bands_with_SF` 0.49 s → 0.015 s, `get_Pmn` 1.6 s → 0.23 s,
  `from_file` RbH12 3×3×3 2.4 s → 1.3 s, `get_bands` RbH12 2.1 s → 0.8 s.
- ✅ `distort_structure` reuses `Structure.expand_structure`.

### Documentation & style
- ✅ NumPy-style docstrings everywhere, with shapes and units; attributes of both
  classes documented in the class docstrings.
- ✅ Consistent imports, type hints, no dead code or commented-out branches.

### Still open for this module
- ⬜ `_alat` is stored in Å by `from_file`/`from_phonopy` but in bohr by
  `change_cell`. Nothing reads it today; pick one unit (or drop the attribute).
- ⬜ `from_phonopy` expects `phonopy_disp.yaml`; the tutorial folder ships
  `phonopy.yaml`. Accept either.
- ⬜ Consider `np.linalg.eigh` (Hermitian) for the dynamical matrix: faster and
  gives real eigenvalues, but changes results at round-off level, so it needs a
  conscious decision.
- ✅ `to_CC` writes its temporary dyn files to a `tempfile` folder, not the cwd.
- ⬜ `get_Pmn` indexes the CellConstructor supercell symmetries (`mapping`,
  `rot_cart`) with `DynQ.lg_idx`, which are indices of the *primitive-cell* spglib
  symmetries. Check that both lists share the same ordering: some characters from
  `compute_traces` on CrSb come out non-integer (e.g. 1.09, 0.248).
- ⬜ Rename `Symmetrize` / `SetupFromSPGLIB` to `symmetrize` / `setup_from_spglib`
  (keep the old names as deprecated aliases).

---

## Stage 1 — Safety net: tests and CI
- ✅ `tests/test_interpolation.py`: Fourier interpolation checked against
  QuantumESPRESSO `matdyn.x` (RbH12 3×3×3, 901 q-points, agreement < 0.01 cm⁻¹),
  grid q-points reproduce the input dyn files in both gauges, `get_dynq` and
  `get_bands` agree, and band paths are sampled correctly for any `N`.
- ✅ GitHub Actions workflow (`.github/workflows/tests.yml`): builds mnx with
  `meson-python` (gfortran + OpenBLAS) and runs the tests on every pull request.
- ⬜ Protect `main` so the `tests` check must pass before merging
  (Settings → Rules / Branches on GitHub).
- ⬜ More tests: `Symmetrize`, `distort_structure`, `write`/`from_file` round trip,
  `from_phonopy`; extend CI to Python 3.11–3.12.
- ⬜ Add `ruff` (lint + format) with a `pyproject.toml` config and a pre-commit hook.

## Stage 2 — `Structure` (`src/mnx/structure.py`)
- ⬜ `Structure.Symmetrize` dominates `DynMatrix.Symmetrize` (≈5 s for CsV3Sb5,
  almost all in `utils.cell.map2structure` → `_fix_coords`, called ~1800 times):
  vectorise it.
- ⬜ Same treatment as `DynMatrix`: docstrings with units, close files, remove the
  bare `except:` in `plot_bz`, fix the "qe" reader assuming
  `ATOMIC_POSITIONS` is the last block of the file.
- ⬜ Vectorise `expand_structure` and the dyn reader.
- ⬜ `from_data` should also set `Nspecies` and masses consistently with the readers.

## Stage 3 — `Bands` (`src/mnx/bands.py`, 1300 lines)
- ⬜ Split into reading, interpolation, Fermi-surface and plotting submodules.
- ⬜ Lazy-import heavy optional dependencies (`pyvista`, `skimage`, `matplotlib`,
  `scipy`) so `import mnx` stays fast and works without a display.
- ⬜ Replace the four bare `except:` blocks with specific exceptions.

## Stage 4 — `utils`
- ⬜ `cell.py`, `classify.py`, `io.py`: docstrings in the same NumPy style,
  `get_rcell` without the per-column `inv` loop, consistent snake_case names.
- ⬜ `consts.py`: cite sources (CODATA) and resolve the "Mass I think this is wrong"
  comment on `Ry2AMU`.

## Stage 5 — Packaging & repository hygiene
- ⬜ `LICENSE` is empty: choose a licence (GPL is the natural match for the bundled
  QuantumESPRESSO Fortran).
- ⬜ `__init__.py`: `__all__ = ["DynMatrix, Structure", "Bands"]` is one string too
  few; should be `["DynMatrix", "Structure", "Bands"]`.
- ⬜ `pyproject.toml`: add the missing runtime deps (`scipy`, `matplotlib`), move
  GUI/3-D extras (`PyQt5`, `pyvista`, `trame-pyvista`) to an optional
  `[project.optional-dependencies] viz` group; read the version from one place.
- ⬜ Stop tracking generated tutorial outputs (`POSCAR_*`, `dyn*` written by the
  notebooks); clear notebook outputs before committing (`nbstripout`).
- ⬜ README: quick-start example, feature list, link to the tutorials, citation.

## Stage 6 — Documentation site
- ⬜ Sphinx + `numpydoc` (or MkDocs + mkdocstrings) API reference built from the
  docstrings, with the tutorial notebooks rendered via `nbsphinx`; publish on
  GitHub Pages from CI.
