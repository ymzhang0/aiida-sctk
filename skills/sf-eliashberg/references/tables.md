# Tables for EPW (deterministic post-processing)

These are code, not judgement: use the functions, do not re-derive them.

- q average: weights of the irreducible q points from the stars of the uniform grid (spglib), then
  `lambda_sf(i nu) = sum_q w_q mu_spin_fluctuation(q, i nu)` and the same for `mu_coulomb`. Written as
  "nu[meV] value" starting at nu = 0, one header line containing the word `imag`.
- DOS file for the full-bandwidth equations: the tetrahedron DOS of the work chain's dos step, re-written in EPW's
  `prefix.dos` format over the whole band (EPW's own file only covers the Fermi window). Check the printed number
  of electrons below E_F: if it is short, the dos step started above the band bottom (MgB2) and must be redone.
- FSM Stoner I: `I = 1/N(0) - 4a` with N(0) the DOS of the m = 0 FSM run itself (same smearing and k grid).
  Using the tetrahedron N(0) with smeared FSM energies gave errors of 0.2-0.8 eV.

Until these are packaged in aiida-sctk / aiida-epw, the scripts are in `scripts/` of this skill.
