# Per-material settings

DFT settings come from the user's own AiiDA export (relaxed lattice, their pseudopotentials); they are inputs, not
convergence parameters. Structures and pseudopotentials are stored in the groups `sf/structures` and `sf/pseudos`
(labels = material / element names; VN uses the pseudopotential labelled `V_vn`).

| Material | ecutwfc / ecutrho (Ry) | smearing | scf k | half-grid shift of the q grid |
| --- | --- | --- | --- | --- |
| V | 84 / 336 | cold 0.02 | 20^3 | (1,1,1) |
| Nb | 82 / 328 | cold 0.02 | 18^3 | (1,1,1) |
| Pb | 54 / 216 | cold 0.02 | 15^3 | (1,1,1) |
| MgB2 | 84 / 336 | cold 0.02 | 16x16x12 | (0,0,1) only |
| HfRuSb, ZrRuSb | 100 / 400 | cold 0.01 | 14^3 | none (unshifted, Gamma skipped) |
| VN (8 atoms) | 80 / 320 | mv 0.02 | 12^3 | none |

## SCTK production parameters (all with nci 12, ci_scale 0.3)

| Material | q grid | nscf k | bands | ecutfock (Ry) | memory per kel job | status of convergence |
| --- | --- | --- | --- | --- | --- | --- |
| V | 8^3 shifted | 32^3 | 20 | 60 | 100 GB | converged; laddxc 0/1/2 scanned |
| Nb | 6^3 shifted | 24^3 | 40 | 30 | 100 GB | converged (RPA scan only) |
| MgB2 | 6^3 shifted (c only) | 24^3 | 80 | 20 | 220 GB | Coulomb kernel not fully converged in q |
| Pb | 6^3 shifted | 24^3 | 40 | 20 | 100 GB | converged (RPA scan only) |
| HfRuSb, ZrRuSb | 6^3 unshifted | 18^3 | 120 | 40 | 220 GB | 120 bands ~3-4 % below the converged value |
| VN | 6^3 unshifted | 12^3 | 120 | 40 | 220 GB | compromise: k and q not converged |

Convergence scans vary one parameter at a time around a small baseline (q 4^3, k 16^3, 20 bands, ecutfock 20, nci 5
for the elements; q 6^3, k 18^3, 40 bands, ecutfock 30, nci 12 for the half-Heuslers). The metric is the q-averaged
lambda_sf(i nu) at 0, 0.1 and 1 eV and the largest eig_max, each changing by less than 5 % per step.

## EPW

| Material | Wannier set | coarse k / q | fine k / q | electron smearing | lambda_ph |
| --- | --- | --- | --- | --- | --- |
| V | 16 WF (s,p,d,f), 40 bands, frozen E_F+4.3 eV (9 WF set for tests) | 12^3 / 6^3 | 60^3 / 20^3 | 0.025 eV | 1.62 |
| Nb | 9 WF (s,p,d), 40 bands, frozen E_F+4.0 eV | 12^3 / 6^3 | 60^3 / 20^3 | 0.025 eV | 1.40 (+-4 %) |
| MgB2 | 12 WF (B s,p; Mg s,p), 40 bands | 12x12x8 / 6x6x4 | 72x72x48 / 24x24x16 | 0.05 eV (0.025 does not converge) | 0.62 |
| Pb | 4 WF (s,p), 30 bands, frozen E_F+3 eV | 12^3 / 6^3 | not converged yet | 0.05 eV | - |
| HfRuSb, ZrRuSb, VN | spectra supplied by the user (files under `sf/export`) | - | - | - | 0.653, 0.889, 1.59 |

Pks of the reference runs are in the record document; do not hard-code them here, they go stale.
