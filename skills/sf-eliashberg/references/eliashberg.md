# Isotropic Eliashberg equations

Inputs (EPW branch epw-wchi): `fila2f` (a2F), `fila2f_sf` (lambda_sf table), `fila2f_c` (mu_C table, replaces `muc`),
`fildos` (wide DOS file, full bandwidth only).

## Four levels, always reported together
1. el-ph only, mu* = 0.13
2. + spin fluctuations, mu* = 0.13
3. + spin fluctuations + mu_C(i nu) from laddxc = 0 (RPA)
4. + spin fluctuations + mu_C(i nu) from laddxc = 2 (KO) - the result the user wants

## Linearised (`tc_linear`, lapack)
Tc = temperature where the largest eigenvalue crosses 1 (interpolate between 1 K steps). Matsubara cutoff 1 eV with
mu*, 2 eV with a Coulomb table (2 and 4 eV agree). It outputs no Z or gap function; for those run the non-linear
constant-DOS solver at one temperature below every Tc.

## Full bandwidth (`full_bandwidth = true`, non-linear)
- `muchem = false`: the chemical-potential update fails on a wide window.
- The solver aborts above Tc instead of returning a zero gap: take Tc from Delta^2(T) extrapolated to zero with a
  0.5 K temperature step.
- The Matsubara cutoff must be several times the DOS window: scan 60 / 120 / 240 eV with sparse sampling
  (`gridsamp = 1`) and report the trend. mu_C is extrapolated as a constant above the last table point (23 eV).
- Still approximate: one Fermi-surface averaged kernel for all energies; fixed chemical potential.

Spectra computed outside this profile are passed as a file; the plugin still needs a parent EPW folder, any finished
a2F run serves (label such runs `a2ffile`).
