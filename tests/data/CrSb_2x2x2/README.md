# CrSb 2×2×2: acoustic sum rule references

References for `tests/test_asr.py`, computed from the dyn files in
[`Tutorial/dyns/CrSb/750`](../../../Tutorial/dyns/CrSb/750) (`hessianv4_1 … hessianv4_4`, 2×2×2
q-grid, 4 irreducible q-points; a metal, so no effective charges).

The q-path is the same as in [`MnTe_2x2x2`](../MnTe_2x2x2): Γ–M–K–Γ–A–L–H–A plus L–Γ,
20 intervals per segment (168 points), as an explicit q-list in crystal coordinates.

| File | Content |
|---|---|
| `hessianv4_0` | grid file needed by `q2r.x` (2×2×2, 4 files); not part of the tutorial data |
| `q2r.in` | QuantumESPRESSO 6.7 `q2r.x` input |
| `matdyn_asr_no.in`, `freq_asr_no` | `matdyn.x` with `asr='no'` and its frequencies (cm⁻¹, 4 decimals) |
| `matdyn_asr_simple.in`, `freq_asr_simple` | same with `asr='simple'` |

To regenerate, copy `Tutorial/dyns/CrSb/750/hessianv4_1 … hessianv4_4` here and run

```bash
q2r.x < q2r.in > q2r.out
matdyn.x < matdyn_asr_no.in > matdyn_asr_no.out
matdyn.x < matdyn_asr_simple.in > matdyn_asr_simple.out
```
