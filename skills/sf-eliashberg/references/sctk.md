# SCTK kernels

Entry point `sctk.spin_fluctuation` (aiida-sctk). Protocol arguments: `pw_code`, `dos_code`, `sctk_code`, `structure`;
codes are `pw-sctk@<computer>`, `dos-sctk@<computer>`, and `sctk-rw@<computer>` for laddxc 0 or `sctk-ko@<computer>`
for laddxc 1 and 2 (the KO build reproduces laddxc 0 point by point, but keep the split so old runs stay comparable).

Inputs to set on top of the protocol:
- `scf`, `nscf`, `twin`: pseudos, cutoffs, smearing (scf only), `nbnd` (nscf and twin); explicit `kpoints` meshes -
  remove `kpoints_distance`.
- `twin_mesh`: the q grid, with offset 0.5 in the shifted directions.
- `sctk.sctk.parameters`: `KEL = {nci, ci_scale, laddxc, lsf: 1, ecutfock, nq1, nq2, nq3}`.
- `qpoints_per_job`: 4 by default; 1 when a job must fit a short time limit.
- kel resources: few MPI processes with many threads is fastest (4 x 16 on 64 cores; `-nk` = number of processes,
  `OMP_STACKSIZE=1G`, `srun -c 16`).

What each step is for: the twin grid (q grid plus its half-shifted copy) supplies the wave functions for the matrix
elements; the dense nscf supplies only the energies for the tetrahedron weights; the dos step is not read by
`sctk.x` (its output is reused later for the full-bandwidth DOS file).

Failures seen: out of memory on q 8^3 or many bands with 100 GB (use 220 GB); nscf task exit code 96 without message
(resubmit); "Nesting occurs" at Gamma (see SKILL.md).

Kernel dump (I(r), f(r) in real space): restart the first kel calculation of a finished run with the code
`sctk-dump@keira` at one q point; the file `sctk_xc_kernel.bin` stays in the work directory.
