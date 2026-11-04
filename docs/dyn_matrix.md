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
- [Polar materials: LO–TO splitting](#polar-materials-loto-splitting)
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
| `phiR` | `[A, B, α, β]` | real-space force constants (short-range part only when `apply_nac`) |
| `zeff` | `[a, α, β]` or `None` | Born effective charges Z*, after the charge sum rule ([LO–TO](#polar-materials-loto-splitting)) |
| `epsilon` | `[α, β]` or `None` | high-frequency dielectric tensor ε∞ |
| `apply_nac` | bool | apply the non-analytic correction; `True` by default when `zeff` and `epsilon` exist |

| Method | Does |
|---|---|
| `from_file(prefix, qgrid, Nqirr, zasr)` | reads QE dyn files `prefix1 … prefixNqirr` (or one file when `Nqirr=0`) |
| `from_phonopy(folder, qgrid)` | reads a VASP + phonopy calculation |
| `get_dynq(q, gauge, direction)` | spectrum at any q, as a one-point `DynQ`; `direction` sets the LO–TO limit at Γ |
| `get_bands(k_inpath, N, gauge)` | dispersion along a path |
| `get_bands_with_SF(k_inpath, N, Q)` | dispersion and one-phonon structure factor at `Q + k` |
| `expand_polvecs(modes, mod)` | polarization vectors and displacements on a supercell |
| `distort_structure(Q, modes, mod)` | supercell displaced along a combination of modes |
| `project_structure(structure, modes, cutoff)` | amplitudes Q of the selected modes in a structure (inverse of `distort_structure`) |
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
| `zeff`, `epsilon` | `[a, α, β]`, `[α, β]` | charges and ε∞ as written in the file (Γ star of a polar material only, else `None`) |
| `direction` | `[3]` or `None` | for a Γ `DynQ` from `get_dynq`: fractional direction used for the LO–TO splitting |

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
| Born effective charges Z* | units of e; `zeff[a, α, β]` = ∂F_aβ/∂E_α (field index first, QE's "E-U" block) |
| Dielectric tensor ε∞ | dimensionless, electronic (clamped-ion) |
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

D(**q**) is Hermitian, so the modes stored in each `DynQ` (from `from_file`, `from_phonopy`
and `Symmetrize`) are computed with `numpy.linalg.eigh` and are **orthonormal**,
⟨e_μ\|e_ν⟩ = δ_μν, also inside degenerate subspaces. Two gauge choices fix their phases:

- at **self-conjugate** q-points (2**q** = **G**, e.g. Γ, M, L), D(**q**) is real in the lattice gauge
  and the eigenvectors are chosen **real**;
- when −**q** (mod **G**) belongs to the same star, its eigenvectors are the **complex conjugates**
  of those at **q**, as allowed by D(−**q**) = D(**q**)*.

Interpolated spectra (`get_dynq`, `get_bands`) come from LAPACK `zheev` in the Fortran routine
`FModules.interpolation.diag_dynq`. Those are orthonormal too, but their phases are not fixed.

### Reading the input

- **QuantumESPRESSO** (`from_file`): Φ(**q**) is read as written in each dyn file (one file per
  star, all q-points of the star). Masses, cell and positions come from the header, with
  either `ibrav = 0` (explicit "Basis vectors") or `ibrav ≠ 0` + `celldm` (built with
  QE's conventions through CellConstructor). The dielectric tensor and the "E-U" effective
  charges are read from the Γ file when present.
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
\mathbf r_A = \mathbf r_A^{0} + \sum_{s}\sum_{i \in \text{star } s} Q_{s,i}\, \mathbf u^{\text{sc}}_{s,i}(A),
$$

with one amplitude Q_{s,i} per q-point of each selected star. Since u has units of
1/√m, Q is in Å·√(Ry mass), so typical amplitudes are tens to hundreds.

The displacement must be **real**, otherwise a `ValueError` is raised:

- at self-conjugate q (Γ, M, L, …) the modes are real, so Q must be real;
- when q and −q are both distorted, their modes are complex conjugates, so Q₋q = Q_q*
  (e.g. `[2, 2]` or `[2+1j, 2-1j]`; `[4, 0]` is rejected).

Every q-point of the selected stars must be commensurate with `mod` (q·mod integer);
otherwise a `ValueError` is raised. Commensurate q-points are snapped to the exact grid values, so
round-off in the input files (e.g. −0.49999999) doesn't add spurious phases.

### Projecting a structure onto modes (`project_structure`)

`project_structure(structure, modes, cutoff=0.2)` is the inverse of `distort_structure`:

```python
distorted = dyn.distort_structure(Q=[[10, 0, 0], [0, 10, 10]], modes=[[2, 0], [3, 0]])
dyn.project_structure(distorted, modes=[[2, 0], [3, 0]])
# [array([10, 0, 0]), array([0, 10, 10])]   (complex dtype)
```

1. **Supercell**: `mod` is read from `structure.cell @ inv(primitive cell)`, which must be a
   diagonal integer matrix (to 10⁻⁴). Strained cells are rejected.
2. **Matching**: every site A of the undistorted supercell (`expand_structure(mod)`) gets the
   atom of the same species at the shortest periodic distance (minimum image). The pairing
   must be one-to-one with all distances below `cutoff` (Å). Otherwise a `ValueError` names
   the failing site or atom. The atom order of `structure` is irrelevant.
3. **Projection**: with Δ**r**_A the minimum-image displacement of the matched atom, each
   amplitude is the independent projection onto its own mode,

$$
Q_{s,i} = \sum_A \sqrt{m_A}\; \mathbf e^{\text{sc}}_{s,i}(A)^{*} \cdot \Delta\mathbf r_A .
$$

Because the modes are orthonormal, the result is **unique**: an amplitude doesn't depend on
which other modes are selected, and a distortion is recovered exactly (to ~10⁻¹³ on the
tests). For any real displacement, modes at self-conjugate q give real amplitudes, and q, −q
pairs give Q₋q = Q_q*.

### Structure factor (`get_bands_with_SF`)

`get_bands_with_SF(k_inpath, N, Q)` evaluates the one-phonon structure factor at the total
momentum transfer **K** = **Q** + **k**, with **Q** a reciprocal lattice vector and **k** running
along the path. It is Eq. (2) of the Supplementary Information of A. Korshunov, M. Alkorta
*et al.*, *Flat band driven competing charge and spin instabilities in the altermagnet CrSb*,
arXiv:2603.25317 (after R. A. Cowley, Rep. Prog. Phys. **31**, 123 (1968), and A. Q. R. Baron,
arXiv:1504.01098), with atomic form factors f_a = 1 and no Debye–Waller factor:

$$
F_\nu(\mathbf K) = \sum_a \frac{\mathbf K \cdot \boldsymbol\varepsilon_{\nu,a}(\mathbf k)}{\sqrt{2 M_a\, \omega_\nu(\mathbf k)}}\,
e^{i \mathbf K \cdot \mathbf R_a},
\qquad S_\nu(\mathbf K) = |F_\nu(\mathbf K)|^2 .
$$

Here **ε**_ν(**k**) are the periodic polarization vectors of QuantumESPRESSO (lattice gauge,
**ε**(**k** + **G**) = **ε**(**k**)) and **R**_a the positions in the primitive cell. With them
*S* depends only on **K**. `mnx` interpolates in the atomic gauge, whose vectors differ by a phase,
**ε**ᵃᵗᵒᵐⁱᶜ_a = e^{−i**k**·**R**_a} **ε**_a, so the code evaluates the same |F| as

$$
F_\nu(\mathbf K) = \sum_a \frac{\mathbf K \cdot \boldsymbol\varepsilon^{\,\mathrm{atomic}}_{\nu,a}(\mathbf k)}{\sqrt{2 M_a\, \omega_\nu(\mathbf k)}}\,
e^{-2\pi i\, \mathbf Q \cdot \mathbf R_a}
$$

(fractional **Q**, cartesian **R**_a). Units are arbitrary: **K** in Å⁻¹ without 2π, M in Ry
atomic units, ω in cm⁻¹. Imaginary modes get *S* = 0. With LO–TO splitting (`apply_nac`), Γ
takes the direction of its segment, as in `get_bands`.

Because *S* depends only on **K**, a path can leave the first Brillouin zone: Γ → M → Γ′ with
**k** from (0, 0, 0) to (1, 0, 0) scans **K** from **Q** to **Q** + (1, 0, 0).

`tests/test_structure_factor.py` checks this formula three ways: against Eq. (2) evaluated
literally on the polarization vectors of the dyn files; that (**k**, **Q**) and
(**k** − **G**, **Q** + **G**) give the same *S*; and, off the grid, against the real normal modes of
4×1×1 and 1×1×3 supercells built from the same force constants, where no phase convention
enters. In space-group-symmetric crystals *S*(**K**) = *S*(R**K**) also holds for every
operation R (checked to 3·10⁻⁷ for CrSb and MnTe).

Earlier versions of `get_bands_with_SF` used √(2M_a)·**u**_a (= √2·**ε**_a, without the
1/√M_a of Eq. (2)) and the phase e^{2πi**K**·**R**_a} on the atomic-gauge vectors, which is not
periodic in **k**. Structure factors computed with earlier versions are not comparable.

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

## Polar materials: LO–TO splitting

In a polar insulator the atoms carry Born effective charges Z*, and a displacement
creates a dipole. Dipoles interact as 1/r³, so the force constants have a long-range
tail that a Fourier interpolation from a coarse grid can't represent. At **q** → 0 the
long-range interaction adds a term that depends on the **direction** of **q**: the
macroscopic electric field of longitudinal optical (LO) modes raises their frequency
above the transverse (TO) ones. That is the **LO–TO splitting**, and it makes the
dispersion discontinuous at Γ.

`mnx` follows X. Gonze and C. Lee, PRB **55**, 10355 (1997), exactly as QuantumESPRESSO's
`q2r.x` / `matdyn.x` (routines `rgd_blk` and `nonanal`).

### Input

QuantumESPRESSO's `ph.x` (with `epsil = .true.`) writes ε∞ and Z* at the end of the Γ dyn
file. `from_file` reads them into `DynQ.epsilon` / `DynQ.zeff` of the Γ star and, if found,
sets `DynMatrix.epsilon`, `DynMatrix.zeff` and `apply_nac = True`. Only the "E-U" block
(Z_{α}{s,β}, field index first) is read, as in `q2r.x`; the "U-E" block is skipped.

**Charge sum rule.** Charge neutrality requires Σ_a Z*_a = 0, which DFPT satisfies only
approximately (MnTe: ~10⁻³ e). `from_file(..., zasr="simple")` (the default) removes the
average, Z*_a ← Z*_a − (1/N) Σ_b Z*_b, as `zasr = 'simple'` in `q2r.x`. `zasr="no"` keeps the
charges as read; `DynQ.zeff` and `write` always keep the raw values.

### Equations

Units are Rydberg atomic units: e² = 2, Ω is the cell volume in bohr³, **q** and **G** are
cartesian and include 2π.

**Long-range (dipole–dipole) term**, an Ewald sum over reciprocal lattice vectors **G**:

$$
\Phi^{\mathrm{LR}}_{ab,\alpha\beta}(\mathbf q) = \frac{4\pi e^2}{\Omega} \Bigg[
\sum_{\mathbf K = \mathbf q + \mathbf G \neq 0}
\frac{(\mathbf K \cdot Z_a)_\alpha (\mathbf K \cdot Z_b)_\beta}{\mathbf K \cdot \varepsilon_\infty \cdot \mathbf K}\,
e^{-\mathbf K \cdot \varepsilon_\infty \cdot \mathbf K / 4\Lambda}\, e^{i \mathbf K \cdot (\boldsymbol\tau_a - \boldsymbol\tau_b)}
\;-\; \delta_{ab} \sum_{\mathbf G \neq 0} \sum_c
\frac{(\mathbf G \cdot Z_a)_\alpha (\mathbf G \cdot Z_c)_\beta}{\mathbf G \cdot \varepsilon_\infty \cdot \mathbf G}\,
e^{-\mathbf G \cdot \varepsilon_\infty \cdot \mathbf G / 4\Lambda} \cos\big(\mathbf G \cdot (\boldsymbol\tau_a - \boldsymbol\tau_c)\big)
\Bigg],
$$

with (**q**·Z_a)_β = Σ_α q_α Z*_{a,αβ}. The second sum makes Φᴸᴿ vanish for a rigid
translation. As in QE, only the reciprocal-space part is kept: the Ewald parameter is
Λ = (2π/alat)² with alat = \|**a**₁\|, and terms with **K**·ε∞·**K** / 4Λ > 14
(e⁻¹⁴ ≈ 10⁻⁶) are dropped. The sum excludes **K** = 0, so Φᴸᴿ is smooth and Φᴸᴿ(Γ) is analytic.
This is the **lattice gauge** (periodic in **q**). In the **atomic gauge** it is multiplied by
e^{i**q**·(**τ**_b − **τ**_a)}, the same phase that relates Φ(**q**) in the two gauges.

**Non-analytic term** at Γ, approached along **q̂**:

$$
\Phi^{\mathrm{NA}}_{ab,\alpha\beta}(\hat{\mathbf q}) = \frac{4\pi e^2}{\Omega}\,
\frac{(\hat{\mathbf q} \cdot Z_a)_\alpha (\hat{\mathbf q} \cdot Z_b)_\beta}{\hat{\mathbf q} \cdot \varepsilon_\infty \cdot \hat{\mathbf q}} .
$$

It is the **K** = **q** → 0 term of the sum above, which is why ω(**q** → 0 along **q̂**) is
continuous with ω(Γ, **q̂**) (`test_loto_limit_is_continuous`).

### Workflow

1. `get_phiR` subtracts Φᴸᴿ(**q**) from every grid point, then Fourier transforms the
   short-range part Φˢᴿ = Φ − Φᴸᴿ. `phiR` holds Φˢᴿ, which decays fast in real space.
2. `get_dynq` / `get_bands` / `get_bands_with_SF` interpolate Φˢᴿ to **q** (Fortran
   `ft_phir`), add Φᴸᴿ(**q**) (Fortran `dipole_dipole`) and, at Γ (or any **G**) with a
   direction, Φᴺᴬ(**q̂**) (Fortran `nonanalytic`), then diagonalize (`diag_dynq`). In the atomic
   gauge Φᴺᴬ gets the same phase e^{2πi**q**·(**τ**_b − **τ**_a)} as the rest of Φ(**q**): it is 1 at **q** = 0
   but not at **q** = **G**, e.g. at Γ′ = (1, 0, 0) of a path into the next zone.

On the grid points this gives back the input dyn files exactly (`test_grid_points_reproduced`).
`Symmetrize` symmetrizes Φˢᴿ and rebuilds the grid matrices with Φᴸᴿ added back.

### Direction at Γ

- `get_dynq([0, 0, 0])` without `direction` gives the **analytic** dynamical matrix (TO
  frequencies, as printed by `ph.x` in the Γ dyn file). `get_dynq([0, 0, 0], direction=u)`
  adds Φᴺᴬ along the fractional direction `u`; the result's `DynQ.direction` records it.
- `get_bands` gives each Γ point the direction of **its own segment**. Since segment ends are
  repeated, a Γ between two segments shows the limit from both sides, e.g. Γ in
  K→Γ→A has the in-plane LO frequency at the end of K→Γ and the z one at the start of Γ→A.
  `matdyn.x` gives the same values on the same list of points (it takes the direction from
  the previous point, or from the next one when the previous point is also Γ).

### Switching it off

```python
dyn = mnx.DynMatrix.from_file("./dyns/MnTe/dyn", qgrid=[2, 2, 2], Nqirr=4)
bands_nac, qpath, xticks = dyn.get_bands(k_inpath, N=50)

dyn.apply_nac = False          # rebuilds phiR from the full Phi(q); the charges are kept
bands_plain, _, _ = dyn.get_bands(k_inpath, N=50)
```

Setting `apply_nac` rebuilds `phiR`. After changing `zeff` or `epsilon` by hand call
`dyn.get_phiR()`.

### Example: MnTe

NiAs-type MnTe (`Tutorial/dyns/MnTe`, 2×2×2 grid) has ε∞ ≈ diag(14.23, 14.23, 14.44)
and Z*(Mn) ≈ +2.82 (in-plane), +2.59 (z). At Γ (cm⁻¹):

| Mode | Analytic (TO) | **q̂** ∥ x (Γ→M) | **q̂** ∥ z (Γ→A) |
|---|---|---|---|
| E₁ᵤ (in-plane dipole) | 134.93 (×2) | 134.93 + **179.83** | 134.93 (×2) |
| A₂ᵤ (dipole along z) | 128.55 | 128.55 | **168.14** |
| Non-polar modes | 93.21, 96.38, 121.13, 180.89 | unchanged | unchanged |

Without the correction the LO modes are wrong near Γ: on this coarse grid the
interpolated bands change by up to 45 cm⁻¹ at **q** = (0.025, 0, 0), next to Γ
(`test_apply_nac_switch`).

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
| `test_loto.py::test_bands_match_quantum_espresso` | QE 6.7 `q2r.x` (`zasr='simple'`) + `matdyn.x` (`asr='no'`), MnTe 2×2×2 with LO–TO, 168 q-points with Γ reached along 4 directions (`tests/data/MnTe_2x2x2`) | < 1e-3 cm⁻¹ (max 1.1e-4) |
| `test_loto.py::test_bands_match_cellconstructor` | CellConstructor `ForceTensor.Tensor2`, same path, `zasr="no"` (`tests/data/MnTe_2x2x2/make_cc_reference.py`) | < 1e-4 cm⁻¹ (max 7e-6) |
| `test_loto.py` (other tests) | charges and ε∞ read and written back; charge sum rule; hexagonal `ibrav=4` cell; grid points reproduced with Φᴸᴿ in both gauges; analytic Γ = `ph.x`; Φᴺᴬ vs an independent NumPy formula; ω(**q** → 0) continuous; gauge phase of Φᴸᴿ; `apply_nac` switch | see the tests |
| `test_structure_factor.py` | Eq. (2) of arXiv:2603.25317 on the dyn-file polarization vectors; invariance under (**k**, **Q**) → (**k** − **G**, **Q** + **G**); real-space 4×1×1 and 1×1×3 supercells off the grid | < 1e-6 (grid), < 1e-9 |
| `test_io.py` | `mnx.io` with real and synthetic bands | exact |
| `test_projection.py` | `distort_structure` → shuffle atoms → `project_structure` returns the input Q; modes orthonormal; integer characters; errors | < 1e-9 |

Run them locally from the repository root with `python -m pytest -v`.

## Known limitations

- **No acoustic sum rule** is imposed on Φ, so acoustic modes at Γ are not exactly zero
  (MnTe: 2.9 and 7.2 cm⁻¹). Compare with `matdyn.x` using `asr='no'`.
- **LO–TO splitting, 3D only**: no 2D Coulomb cutoff (QE's `loto_2d`) for layers in vacuum.
- **Effective charges are not symmetrized**: `Symmetrize` acts on Φˢᴿ only; ε∞ and Z* are
  used as read (plus the charge sum rule).
- **phonopy**: `from_phonopy` does not read `BORN` files yet, so phonopy inputs get no
  non-analytic correction.
- **Frequency conversion**: `consts.Ry2cm` differs from QE's Ry → cm⁻¹ factor by 4·10⁻⁷
  (relative), visible only in the last printed digits.
- **`project_structure` memory**: atom matching builds an Nₐₜ × Nₐₜ distance table, which is
  fine up to a few thousand atoms in the supercell.
- **`_alat` units**: Å after `from_file` / `from_phonopy`, bohr after `change_cell`.
  Nothing reads it today.
- **phonopy deprecations**: phonopy ≥ 3.x warns that `get_dynamical_matrix_at_q` is
  deprecated. It still works and is kept for compatibility with older phonopy.
