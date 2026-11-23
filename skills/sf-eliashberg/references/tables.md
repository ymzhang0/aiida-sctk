# Tables for EPW (deterministic post-processing)

These are code, not judgement: call the tools, do not re-derive the algorithms. They live in `aiida_sctk.tools`
and are exposed to aiida-mcp (entry-point group `aiida_mcp.tools`); they read nodes and write files, nothing is
stored in the provenance.

| Tool | What it returns |
| --- | --- |
| `average_sctk_kernels(pk)` | q-averaged lambda_sf and mu_C at 0 / 0.1 / 1 eV, largest and averaged eig_max, fraction of the q grid covered - the convergence metric |
| `write_epw_kernel_tables(pk, directory, name)` | writes `<name>.lam_sf_iw` (fila2f_sf), `<name>.muc_iw` (fila2f_c) and `<name>.dos_fbw` (fildos) |
| `get_fsm_stoner_parameter(group)` | Stoner I from the FSM runs of a group, with N(0) taken from the m = 0 run itself |

How they work, so that the numbers can be trusted and checked:
- q average: weights of the irreducible q points from the stars of the uniform grid (spglib), then
  `lambda_sf(i nu) = sum_q w_q mu_spin_fluctuation(q, i nu)`, the same for `mu_coulomb`. Table format: one header
  line containing the word `imag`, then "nu[meV] value" starting at nu = 0.
- DOS file: the tetrahedron DOS of the work chain's dos step in EPW's `prefix.dos` format over the whole band
  (EPW's own file only covers the Fermi window). Check `electrons_below_fermi_in_window`: if it is below the valence
  count, the dos step started above the band bottom (MgB2) and must be redone before a full-bandwidth run.
- FSM: `I = 1/N(0) - 4a`. Using a tetrahedron N(0) with smeared FSM energies gave errors of 0.2-0.8 eV.

The commensurate-grid rule is enforced by the work chain itself (input validator), so a wrong pair is refused at
submission. The plotting scripts are in `scripts/` of this skill; the a2F / Wannier / Tc summaries are still scripts
in `~/aiida/sf-run` until they move into aiida-epw.
