# EPW electron-phonon spectrum

1. `epw.epw_prep`: scf -> nscf -> wannier90 -> ph -> EPW coarse (epmatwp).
2. **Check the Wannier set before anything else**: interpolated bands against a DFT bands run on the same path;
   compare max and rms deviation within E_F +- 0.4 eV. V reaches 3 meV, Pb only 15 meV.
3. Fine-grid a2F run (restart from the coarse folder). Settings that matter:
   - `etf_mem = 0`; the default re-reads the epmatwp file for every q (60 s per q for 9 GB).
   - pools limited by epmatwp size x processes <= memory.
   - fine k >= 48^3 for V (24^3 overestimates the DOS by 40 %); fine q must divide fine k; k 48 with q 16 fails
     (`kpmq_map`), use 60 / 20.
   - electron smearing 0.025 eV for V and Nb, 0.05 eV for MgB2 and Pb. If the printed DOS jumps between grids,
     the smearing is too small.
4. Convergence table: DOS per spin, lambda_ph, omega_log, Allen-Dynes Tc for each grid; quote the spread of the two
   densest grids as the uncertainty.

The plugin reserves `liso/limag/laniso/fbw/lreal/lpade/lacon`: use the inputs `momentum_dependence`, `real_axis`,
`full_bandwidth` instead of writing them into the parameters.
