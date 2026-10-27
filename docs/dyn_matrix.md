# Phonons with `DynMatrix`

`mnx.DynMatrix` (in [`src/mnx/dyn_matrix.py`](../src/mnx/dyn_matrix.py)) holds the phonon
problem of a crystal: the dynamical matrices on a regular q-grid, the real-space force
constants obtained from them, and tools to interpolate, plot, distort and symmetrize.
The plotting helpers live in `mnx.io` ([`src/mnx/utils/io.py`](../src/mnx/utils/io.py)).

This page covers what each part does, the units and conventions, and the equations
behind them. The notebook [`Tutorial/DynMatrix.ipynb`](../Tutorial/DynMatrix.ipynb) walks
through the same features on real data.

- [Quick start](#quick-start)
- [Objects and data layout](#objects-and-data-layout)
- [Units and conventions](#units-and-conventions)
- [Equations](#equations)
- [Plotting with `mnx.io`](#plotting-with-mnxio)
- [Validation](#validation)
- [Known limitations](#known-limitations)

---

## Quick start

```python
import matplotlib.pyplot as plt
import mnx

# QuantumESPRESSO: hessian1 ... hessian6 computed on a 3x3x3 grid
dyn = mnx.DynMatrix.from_file("./dyns/RbH12/3x3x3/hessian", qgrid=[3, 3, 3], Nqirr=6)
# or VASP + phonopy: primPOSCAR/POSCAR, phonopy_disp.yaml/phonopy.yaml and FORCE_SETS
# dyn = mnx.DynMatrix.from_phonopy("./dyns/phonopy", qgrid=[2, 2, 2])

dyn.info()

k_inpath = [
    [[0, 0, 0], [0.5, 0, 0]],          # Gamma -> M
    [[0.5, 0, 0], [1/3, 1/3, 0]],      # M -> K
    [[1/3, 1/3, 0], [0, 0, 0]],        # K -> Gamma
]
bands, qpath, xticks = dyn.get_bands(k_inpath, N=50)

fig, ax = plt.subplots()
mnx.io.plot_bands(ax, qpath, bands, xticks, [r"$\Gamma$", "M", "K", r"$\Gamma$"], label="RbH12")
ax.legend()
```

## Objects and data layout

A `DynMatrix` contains one `DynQ` per irreducible q-star. Every array uses the index
names below.

| Index | Range | Meaning |
|---|---|---|
| `s` | `Nqirr` | irreducible q-star |
| `qi` | `Nqinstar` | q-point inside a star |
| `a`, `b` | `Natoms` | atom of the primitive cell |
| `A`, `B` | `Natoms·∏qgrid` | atom of the supercell |
| `α`, `β` | 3 | cartesian direction |
| `ν` (mode) | `3·Natoms` | phonon branch, sorted by increasing ω² |

### `DynMatrix`

| Attribute | Shape | Content |
|---|---|---|
| `structure` | – | primitive cell (`mnx.Structure`) |
| `super_structure` | – | supercell commensurate with `qgrid` |
| `qgrid` | `[3]` | q-grid of the calculation |
| `Nqirr`, `Nqpoints` | – | number of stars and of q-points in the grid |
| `DynQs` | `[Nqirr]` | one `DynQ` per star |
| `qstars` | `[Nqirr]` of `[Nqinstar, 3]` | fractional q-points of each star |
| `phiR` | `[A, B, α, β]` | real-space force constants |

| Method | Does |
|---|---|
| `from_file(prefix, qgrid, Nqirr)` | reads QE dyn files `prefix1 … prefixNqirr` (or one file when `Nqirr=0`) |
| `from_phonopy(folder, qgrid)` | reads a VASP + phonopy calculation |
| `get_dynq(q, gauge)` | spectrum at any q, as a one-point `DynQ` |
| `get_bands(k_inpath, N, gauge)` | dispersion along a path |
| `get_bands_with_SF(k_inpath, N, Q)` | dispersion and one-phonon structure factor at `Q + k` |
| `expand_polvecs(modes, mod)` | polarization vectors and displacements on a supercell |
| `distort_structure(Q, modes, mod)` | supercell displaced along a combination of modes |
| `Symmetrize(symprec)` | symmetrizes structure and force constants |
| `get_Pmn(symprec)` | little-group representation matrices of every star |
| `write(prefix)` / `to_CC()` | QE dyn files / CellConstructor `Phonons` |
| `change_cell(cell)`, `copy()`, `info()` | utilities |

### `DynQ`

| Attribute | Shape | Content |
|---|---|---|
| `qpoints`, `qpoints_cart` | `[qi, 3]` | fractional / cartesian q-points |
| `phiqs` | `[qi, a, b, α, β]` | force constants Φ(q) |
| `dynqs` | `[qi, a, b, α, β]` | dynamical matrices D(q) |
| `frequencies` | `[qi, ν]` complex | ω in cm⁻¹ |
| `polvecs` | `[qi, ν, 3a+α]` | eigenvectors e (rows) |
| `displacements` | `[qi, ν, 3a+α]` | e / √m |
| `degs` | `[ν, ν]` bool | degenerate pairs (\|Δω\| < 10⁻³ cm⁻¹) |
| `lg_idx` | list | spglib indices of the little group of the star |
| `Pmn`, `traces` | `[ν, ν, S]`, `[ν, S]` | set by `get_Pmn` / `compute_traces` |

## Units and conventions

| Quantity | Unit / convention |
|---|---|
| Lengths, `cell`, `atom_coords` | Å; lattice vectors are the **rows** of `cell` |
| Reciprocal cell `rcell` | rows **b**ᵢ with **a**ᵢ·**b**ⱼ = δᵢⱼ, i.e. **without 2π**, Å⁻¹ |
| q-points `qpoints`, `k_inpath` | fractional, **q** = Σᵢ qᵢ **b**ᵢ |
| Cartesian q `qpoints_cart` | Å⁻¹ without 2π. QE writes q in 2π/alat; mnx stores q_QE / alat |
| Masses | Rydberg atomic units: m = m[amu] × 911.444 (as in QE dyn files) |
| Force constants Φ | Ry/bohr² |
| Eigenvalues λ = ω² | Ry² (Rydberg atomic units, ħ = 1) |
| Frequencies | cm⁻¹: ω = √λ × 109 737.36 (`consts.Ry2cm`); THz = cm⁻¹ × 0.0299792458 |
| Imaginary modes | stored as complex ω = i\|ω\|; `get_bands` and `write` report them as −\|ω\| |
| Band-path distance `qpath` | 10 × cumulative \|Δ**k**\| in Å⁻¹ without 2π, i.e. nm⁻¹ |
| Supercell atom order | cell (i, j, k) with k fastest: `A = Natoms·(m₂m₃·i + m₃·j + k) + a` |
| Supercell vector component | `3·A + α` |

## Equations

### Dynamical matrix and phonons

The dynamical matrix is the mass-scaled force constant matrix

$$
D_{ab}^{\alpha\beta}(\mathbf q) = \frac{\Phi_{ab}^{\alpha\beta}(\mathbf q)}{\sqrt{m_a m_b}} ,
$$

flattened to a `3N × 3N` matrix with index `3a + α`. Its eigenpairs are the squared
frequencies and the polarization vectors

$$
D(\mathbf q)\, \mathbf e_\nu(\mathbf q) = \omega_\nu^2(\mathbf q)\, \mathbf e_\nu(\mathbf q),
\qquad \omega_\nu = \sqrt{\omega_\nu^2}\ \ (\text{complex square root}).
$$

Modes are sorted by increasing ω². The atomic displacement pattern of a mode is

$$
u_{\nu, a}^{\alpha} = \frac{e_{\nu, a}^{\alpha}}{\sqrt{m_a}} .
$$

`DynQ.from_file` diagonalizes with `numpy.linalg.eig`. Interpolated spectra (`get_dynq`,
`get_bands`) come from LAPACK `zheev` in the Fortran routine `FModules.interpolation.diag_dynq`.

### Reading the input

- **QuantumESPRESSO** (`from_file`): Φ(**q**) is read as written in each dyn file (one file per
  star, all q-points of the star). Masses, cell and positions come from the header.
- **phonopy** (`from_phonopy`): phonopy's D(**q**) in eV/Å²/amu is evaluated at every point of
  `qgrid` and converted to Φ(**q**) = D(**q**)·√(m_a m_b) in Ry/bohr² (× 0.0205817 × 2/1822.89).
  Every grid point is stored as a star of its own.

### Fourier transforms and gauges

From the grid to real space (`get_phiR`, Fortran `ift_fcq`):

$$
\Phi_{AB}^{\alpha\beta} = \frac{1}{N_q} \sum_{\mathbf q \in \text{grid}}
\mathrm{Re}\left[\Phi_{ab}^{\alpha\beta}(\mathbf q)\, e^{-2\pi i\, \mathbf q \cdot \mathbf R_{AB}}\right],
$$

where `a`, `b` are the primitive-cell images of supercell atoms `A`, `B`, and the
fractional vector **R**_AB depends on the **gauge**:

| Gauge | **R**_AB | Phase convention of Φ(**q**) |
|---|---|---|
| `"lattice"` | **T**_B − **T**_A, between the cells containing A and B | QuantumESPRESSO dyn files |
| `"atomic"` | **r**_B − **r**_A, between the atoms | includes the intra-cell positions |

`from_file` transforms QE dyn files with the lattice gauge. `from_phonopy` uses the
atomic gauge, because phonopy's D(**q**) includes the atomic positions in its phases. Both gauges give the same eigenvalues and
differ only by atom-dependent phases in the eigenvectors.

Back to any **q** (`get_dynq`, `get_bands`, Fortran `ft_fcr`), with **Wigner–Seitz
folding**:

$$
\Phi_{ab}^{\alpha\beta}(\mathbf q) = \sum_{\mathbf T} \Phi_{a, b+\mathbf T}^{\alpha\beta}\;
\frac{1}{n_{\mathbf T}} \sum_{\mathbf S \in \mathcal W_{\mathbf T}} e^{2\pi i\, \mathbf q \cdot (\mathbf R + \mathbf S)} .
$$

Here **T** runs over the cells of the supercell and **R** is the gauge vector above. 𝒲
is the set of shortest images of **R** among the 27 shifts **S** = (n₁m₁, n₂m₂, n₃m₃),
nᵢ ∈ {−1, 0, 1} (at most 8, equal within 10⁻⁵ Å), and n_T is their number.

In the **atomic gauge** the shortest image is chosen by atom-to-atom distance, as in
QuantumESPRESSO's `matdyn.x` and in phonopy. That's why it reproduces both (see
[Validation](#validation)). In the **lattice gauge** it is chosen by cell-to-cell
distance, which differs from them off the grid by up to ~20 cm⁻¹ on the test data.
At grid q-points both gauges reproduce the input exactly. **Use `gauge="atomic"` (the
default) for band structures.**

### Band paths (`get_bands`)

Each segment [**k**_s, **k**_e] of `k_inpath` is sampled with `N + 1` points
**k**(t) = **k**_s + t(**k**_e − **k**_s), t = 0, 1/N, …, 1. Joints are therefore repeated.
`qpath` accumulates 10·\|Δ**k**\|_cart, and `xticks` holds its value at the start and at every
segment end.

### Supercell expansion and distortions

A mode at **q** is expanded to the supercell `mod` by the Bloch phase of each cell
**T** = (i, j, k):

$$
e^{\text{sc}}_{\nu}(\mathbf T, a) = \frac{1}{\mathcal N}\, e_{\nu, a}(\mathbf q)\, e^{2\pi i\, \mathbf q \cdot \mathbf T},
\qquad u^{\text{sc}} = e^{\text{sc}} / \sqrt{m},
$$

where 𝒩 normalizes e^sc over the whole supercell. `distort_structure(Q, modes, mod)`
displaces the supercell along

$$
\mathbf r_A = \mathbf r_A^{0} + \sum_{s}\sum_{i \in \text{star } s} \mathrm{Re}\left[Q_{s,i}\, \mathbf u^{\text{sc}}_{s,i}(A)\right],
$$

with one amplitude Q_{s,i} per q-point of each selected star. Since u has units of
1/√m, Q is in Å·√(Ry mass), so typical amplitudes are tens to hundreds.

### Structure factor (`get_bands_with_SF`)

For **K** = **k** + **Q** in cartesian units (Å⁻¹ without 2π) and each mode,

$$
S_\nu(\mathbf K) = \frac{1}{\omega_\nu}\left|\sum_a \sqrt{2 m_a}\; e^{2\pi i\, \mathbf K \cdot \mathbf r_a}\; \mathbf u_{\nu,a} \cdot \mathbf K\right|^2 ,
$$

with ω in cm⁻¹ and the atomic gauge. As implemented, √(2m_a)·**u**_a = √2·**e**_a, so the
masses cancel. No scattering lengths or form factors are included.

### Symmetrization (`Symmetrize`)

1. The structure is symmetrized with spglib (`Structure.Symmetrize`), and the supercell
   is rebuilt.
2. Permutation symmetry: Φ_AB^αβ ← ½(Φ_AB^αβ + Φ_BA^βα).
3. Invariance under the pure translations of the supercell, then under the space
   group. Both use QuantumESPRESSO's routines (`symph.trans_v2`, `symph.sym_v2`).
4. Φ(**q**), frequencies and eigenvectors of every star are recomputed from the
   symmetrized Φ_AB (lattice gauge).

### Little-group representations (`get_Pmn`, `compute_traces`)

For every star, using its first q-point expanded to the q-grid supercell, and every
operation S = {R|**t**} of the little group:

$$
P_{\mu\nu}(S) = \sum_{A} \mathbf e_{\mu}^{\text{sc}}(A)^{*} \cdot R\, \mathbf e_{\nu}^{\text{sc}}(S^{-1}A),
$$

kept only for degenerate pairs (μ, ν) and zero otherwise. `compute_traces` gives the
character of each mode, i.e. the trace over its degenerate block:
χ_μ(S) = Σ_{ν ∈ deg(μ)} P_νν(S).

## Plotting with `mnx.io`

All helpers take the outputs of `get_bands` / `get_bands_with_SF` unchanged, draw on a
matplotlib `Axes` and return it, so several data sets can share a panel.

| Function | Input | Draws |
|---|---|---|
| `plot_bands(ax, qpath, bands, xticks, xlabels, color, lw, alpha, label, ls)` | `get_bands` | every band as a line; `label` goes to the legend once |
| `plot_bands_segment(qpath, bands, data, xticks, xlabels, ax=None, vmin, vmax, cmap, num_interp)` | `get_bands_with_SF` | bands coloured (and faded) by `data` on a log scale, with a colour bar |
| `map_from_bands(ax, qpath, bands, xticks, xlabels, data, emin, emax, …, sigma, …)` | `get_bands_with_SF` | intensity map I(x, E) = Σ_ν w_ν(x) exp[−(E − ω_ν(x))²/2σ²], bands on top |

`map_from_bands` also returns the map `[Ny, Nx_uniform]` and its axes `grid_x`, `grid_y`.
The Gaussian isn't normalized, so a band of weight w peaks at w. All three accept any
number of bands, and the energy unit is whatever `bands` uses (cm⁻¹ for phonons,
eV for electrons from `mnx.Bands`).

**Comparing two calculations**

```python
bands_750, qpath, xticks = dyn750.get_bands(k_inpath, N=20)
bands_500, _, _ = dyn500.get_bands(k_inpath, N=20)

fig, ax = plt.subplots()
mnx.io.plot_bands(ax, qpath, bands_750, xticks, xlabels, label="T = 750 K")
mnx.io.plot_bands(ax, qpath, bands_500, xticks, xlabels, color="red", label="T = 500 K")
ax.legend()
```

**Structure factor along a path**

```python
bands, qpath, xticks, sf = dyn.get_bands_with_SF(k_inpath, N=100, Q=[0, 0, 3])

ax = mnx.io.plot_bands_segment(qpath, bands, sf, xticks, xlabels)  # coloured lines

fig, ax = plt.subplots()
ax, mmap, grid_x, grid_y = mnx.io.map_from_bands(
    ax, qpath, bands, xticks, xlabels, sf, emin=0, emax=300, sigma=1, vmax=1e-1
)
```

## Validation

The tests in [`tests/`](../tests) run on every pull request (`.github/workflows/tests.yml`):

| Test | Reference | Agreement |
|---|---|---|
| `test_interpolation.py::test_bands_match_quantum_espresso` | QE `matdyn.x`, RbH12 3×3×3, 901 q-points (inputs and output in `tests/data/RbH12_3x3x3`) | < 0.01 cm⁻¹ (max 0.002) |
| `test_phonopy.py::test_interpolation_matches_phonopy` | phonopy, on and off the 2×2×2 grid | < 0.01 cm⁻¹ (max 1e-4) |
| `test_interpolation.py::test_grid_points_reproduce_dyn_files` | input dyn files, both gauges | < 1e-3 cm⁻¹ |
| `test_io.py` | `mnx.io` with real and synthetic bands | exact |

Run them locally from the repository root with `python -m pytest -v`.

## Known limitations

- **Single precision phases**: the Fortran interpolation stores the Fourier phases as
  single-precision `complex` (`phase`, `phase_weight` in `interpolation.f90`). That limits
  the relative accuracy to ~10⁻⁷, which is still well below the QE comparison tolerance.
- **No non-analytic correction**: there is no LO–TO splitting term for polar materials, and
  no acoustic sum rule is imposed on Φ.
- **`get_Pmn` index check**: the supercell symmetries from CellConstructor are indexed with
  `lg_idx`, which comes from the primitive-cell spglib list. Some characters come out
  non-integer on CrSb, so check that both lists are ordered the same way.
- **`_alat` units**: Å after `from_file` / `from_phonopy`, bohr after `change_cell`.
  Nothing reads it today.
- **phonopy deprecations**: phonopy ≥ 3.x warns that `get_dynamical_matrix_at_q` is
  deprecated. It still works and is kept for compatibility with older phonopy.
