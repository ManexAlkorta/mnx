# MnTe 2×2×2: LO–TO splitting references

References for `tests/test_loto.py`, computed from the dyn files in
[`Tutorial/dyns/MnTe`](../../../Tutorial/dyns/MnTe) (NiAs-type AFM MnTe, PBE+U, 2×2×2 q-grid,
4 irreducible q-points; `dyn1` holds ε∞ and the Born effective charges).

The q-path is Γ–M–K–Γ–A–L–H–A plus L–Γ, 20 intervals per segment (168 points, segment
ends repeated). At each Γ the LO–TO direction is that of its own segment, so Γ is reached
along x (Γ–M), along K–Γ, along z (Γ–A) and along L–Γ.

| File | Content |
|---|---|
| `q2r.in`, `matdyn.in` | QuantumESPRESSO 6.7 inputs: `zasr='simple'`, `asr='no'`, explicit q-list in crystal coordinates |
| `freq` | `matdyn.x` output (`flfrq`), cm⁻¹ with 4 decimals |
| `matdyn_asr_simple.in`, `freq_asr_simple` | same with the acoustic sum rule `asr='simple'` (`tests/test_asr.py`) |
| `make_cc_reference.py` | script that writes `cc_freq.txt` with CellConstructor's `ForceTensor.Tensor2` |
| `cc_freq.txt` | CellConstructor frequencies, charges without sum rule (`zasr="no"` in mnx) |

To regenerate the QuantumESPRESSO reference, copy `Tutorial/dyns/MnTe/dyn0 ... dyn4` here and run

```bash
q2r.x < q2r.in > q2r.out
matdyn.x < matdyn.in > matdyn.out
matdyn.x < matdyn_asr_simple.in > matdyn_asr_simple.out
```

`matdyn.x` takes the LO–TO direction at q = 0 from the previous point of the list, or from the
next one when the previous point is also Γ or there is none, which gives the same directions
as mnx's per-segment rule on this path.
