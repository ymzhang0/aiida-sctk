---
name: sf-eliashberg
description: Run and analyse the spin-fluctuation superconductivity workflow of this project through AiiDA - SCTK kernels (spin fluctuation, screened Coulomb with laddxc 0/1/2), EPW electron-phonon spectra, and isotropic Eliashberg Tc at the four levels of approximation (el-ph, + SF, + RPA Coulomb, + KO Coulomb), constant-DOS and full-bandwidth. Use whenever the user asks to run, converge, check or plot anything for V, Nb, VN, MgB2, Pb, HfRuSb, ZrRuSb or a new material in this project, mentions SCTK, kel, lambda_sf, mu_C, laddxc, KO interaction, a2F, fila2f_sf / fila2f_c, full bandwidth, production kernels or convergence tests, or asks to update the record document - even if they do not name the workflow.
---

# Spin-fluctuation Eliashberg workflow

The deliverable of this project is the Eliashberg solution from `epw.x`. Everything is submitted through AiiDA
(profile `sf`) with the aiida MCP tools; this skill supplies what those tools cannot know: which parameters to use,
in what order the steps go, which rules must hold, and how results are written up. Read the reference file for the
step you are on rather than all of them.

## The chain

1. **SCTK kernels** (`sctk.spin_fluctuation`): scf -> dense nscf -> twin grid -> dos -> kel jobs split over q -> merge.
   Output: per-q `eig_max`, `mu_spin_fluctuation`, `mu_coulomb` on an imaginary-frequency grid. See `references/sctk.md`.
2. **Tables for EPW**: star-weighted q average -> `lambda_sf(i nu)`, `mu_C(i nu)`; tetrahedron DOS over the whole band.
   See `references/tables.md`.
3. **EPW spectrum** (`epw.epw_prep`, then a fine-grid a2F run): Wannier quality first, then fine-grid convergence.
   See `references/epw.md`.
4. **Isotropic Eliashberg**: four levels of approximation, linearised (Tc from the largest eigenvalue) and
   full bandwidth (non-linear, DOS over the whole band). See `references/eliashberg.md`.
5. **Record document**: every result goes into its material's section in the fixed layout. See `references/record-doc.md`.

Per-material settings (cutoffs, grids, bands, memory, pks of the reference runs) are in `references/materials.md`;
cluster options (keira, LUMI) in `references/clusters.md`.

## Rules that are easy to break

- **Commensurate grids.** `sctk.x` takes the band energies at k+q from the dense nscf grid by rounding, so the nscf
  k grid must be a multiple of the q grid (twice that in a direction with a half-grid shift). A non-commensurate
  pair runs without error and gives wrong kernels.
- **`laddxc` only changes the charge channel** (0 = RPA, 1 = one-sided f, 2 = Kukkonen-Overhauser). The spin-fluctuation
  kernel is identical for the three; if it is not, something else changed between the runs.
- **Never set `lsf = 0`**, and never edit the pristine `SCTK/` source tree; the modified copies are `SCTK_rw`, `SCTK_ko`, `SCTK_dump`.
- **The KO result needs production-level kernels with f** (`laddxc = 2`) plus the spin-fluctuation kernel, solved
  with `mu_C(i nu)` and no `mu*`. A Tc obtained with `mu* = 0.13` is not the KO result; label it as what it is.
- **Unshifted q grids contain Gamma as q point 1**, which `sctk.x` rejects ("Nesting occurs"): the work chain skips it;
  a single-q restart (e.g. the kernel dump) must start from q point 2.
- **Post-processing stays outside the provenance**: the analysis scripts read nodes and write tables and figures;
  do not turn them into calcfunctions.
- **Ideas discussed are not instructions.** Change code or submit runs only for what the user asked to be done.

## Before submitting

State the material, the entry point, the parameters that differ from `references/materials.md`, the cluster and the
resources, and confirm unless the user already did. Put runs in the group `sf/sctk/<material>` or `sf/epw/<material>`
and give each a label `<material>:<what changed>` - the tables and figures find runs by these labels.

## Reporting

Give numbers with the pk they come from. Say what is converged and to how many percent, and what is not; say when a
value is an extrapolation or an estimate. A calculation that is still queued is reported as queued, not predicted.
