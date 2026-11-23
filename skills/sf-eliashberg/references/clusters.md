# Clusters

| | keira (Manneback) | LUMI |
| --- | --- | --- |
| AiiDA computer | `keira` (local transport) | `lumi` (SSH, safe interval 60 s) |
| partition | `keira`; `Def` only when free | `standard` (whole nodes, 128 cores, no memory request); `debug` (30 min, 2 jobs) for validation only |
| account | - | `project_465002926` |
| kel job | 4 MPI x 16 threads, 100-220 GB | 8 MPI x 16 threads |
| codes | pw-sctk, dos-sctk, ph-sctk, sctk-rw, sctk-ko, sctk-dump, epw-wchi | pw-sctk, dos-sctk, sctk-rw, sctk-ko |

- Builds and heavy runs go through batch jobs, never on a login node; keep polling intervals at 10 minutes or more.
- A work chain runs on one cluster from scf to the end; restarts cannot cross clusters, so EPW and Eliashberg runs
  stay where their parent folders are.
- A new code version on a second cluster is validated first: the small V test (q 4^3 shifted, k 16^3, 20 bands)
  must reproduce eig_max, lambda_sf and mu_C of the reference run point by point.
