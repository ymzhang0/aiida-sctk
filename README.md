# aiida-sctk

AiiDA plugin for the patched QE/SCTK `calculation='kel'` polarizability workflow.
Structure follows the sibling aiida-triqs package (src layout, Hatch, entry points).

## Install

```sh
python -m pip install -e '.[tests]'
pytest -q
```

Entry points:
- CalculationFactory('sctk.kel'): SCTKCalculation
- ParserFactory('sctk.kel'): SCTKParser
- WorkflowFactory('sctk.base'): SCTKBaseWorkChain
- WorkflowFactory('sctk.polarizability'): PolarizabilityWorkChain

## Data flow

SCF → dense NSCF → DOS → twin bands → kel.
SCF, NSCF and twin use QE PwBaseWorkChain. SCTK uses SCTKBaseWorkChain,
following aiida-qe's BaseRestartWorkChain pattern. DOS uses the official DosCalculation:
installed aiida-qe does not expose a standalone DOS BaseWorkChain.
NSCF automatically enables SYSTEM.la2f and tetrahedra_opt. The dense NSCF
out/aiida.a2Fsave is copied separately into kel's out/ directory. Twin uses SCF
charge density and explicit double-grid k points. The twin out/ directory is
copied privately into kel, keeping aiida.save and wavefunctions. No ancestor
folder is modified. Intermediate remote folders are retained for reproducibility.

`twingrid.x` is a shell script: the calcfunction reproduces its loop order,
unit weights and both grids, including duplicate points when shifts are zero.
`twin_mesh` specifies this wavefunction grid; KEL.nq1/nq2/nq3 independently
specify the q grid. The mesh offset is converted to CONTROL.lshift_q.

SCTK owns CONTROL.prefix='aiida', outdir='./out/', calculation='kel', and
restart_mode='from_scratch'. Do not include these in parameters. Each AiiDA job
starts in a fresh directory, so appended head files from previous kel jobs are
not reused. Prefix changes from the shell example's pwscf to aiida throughout.

## Builder

See examples/builder.py. It constructs and validates a builder but does not
submit. Supply registered code labels and an existing StructureData PK plus a
kind-name → UpfData PK mapping. Use pw.x, dos.x and sctk.x from the same patched
QE/SCTK build on the same computer; stock QE may not generate a2Fsave.

The provided TiRuSb example uses ecutwfc=84, ecutrho=336, nbnd=40, dense 18³,
twin 4³ and q 4³. Adapt all physical parameters and pseudopotentials to the system.
NSCF and twin must use matching band counts, spin settings and pseudopotentials;
SCTK's dense eigenvalue data and twin wavefunctions must be consistent. Disable
PW symmetry-changing/custom k-point settings that would reorder the twin list.
Do not use disk_io='none' or clean the PW folders before kel completes.

Scheduler options are set per step. MPI pool count must satisfy SCTK's
npool × band-groups = MPI ranks constraint (32 ranks, -nk 32 in the example).
Configure the computer MPI launcher for srun; AiiDA writes the scheduler script.
The LUMI example uses 32 MPI ranks × 4 OpenMP threads for kel, 128 × 1 for PW,
and 1 × 1 for DOS. Module setup belongs in the code/computer prepend text.

## Outputs and limits

`sctk.polarizability` is the exact aiida.pi0_head.dat file produced by the user's
patched SCTK. `sctk.ixc_polarizability` is optional. stdout and both head files
are retrieved; out/aiida.vel* remain in sctk.remote_folder. DOS outputs appear
under dos. No unsupported interpretation of head-file columns is imposed.
The parser requires JOB DONE and pi0_head; unpatched SCTK without this diagnostic
output is deliberately reported as missing polarizability. This is a fresh-run
workflow, not an automatic kel checkpoint-restart workflow.

Tests cover registration, input serialization, remote-copy layout, double-grid
order and provenance. Real LUMI calculations are required to validate numerical
results and the compiled executables; no production calculations are launched
by the tests or builder example.

## Protocols

Both workflows implement the aiida-qe `ProtocolMixin` interface and provide
`fast`, `balanced` (default), and `stringent` protocols. Inspect them with
`get_available_protocols()` or `get_protocol_inputs()`. Construct builders with:

```python
base = WorkflowFactory('sctk.base').get_builder_from_protocol(
    sctk_code, twin_parent, dense_parent, protocol='balanced')
workflow = WorkflowFactory('sctk.polarizability').get_builder_from_protocol(
    pw_code, dos_code, sctk_code, structure, protocol='balanced', overrides={
        'scf': {'pw': {'pseudos': pseudos,
                       'parameters': {'SYSTEM': {'ecutwfc': 84., 'ecutrho': 336.}}}},
        'nscf': {'pw': {'pseudos': pseudos,
                        'parameters': {'SYSTEM': {'ecutwfc': 84., 'ecutrho': 336., 'nbnd': 40}}}},
        'twin': {'pw': {'pseudos': pseudos,
                        'parameters': {'SYSTEM': {'ecutwfc': 84., 'ecutrho': 336., 'nbnd': 40}}}},
    })
```

As in aiida-qe, the PW protocols use an installed pseudo family unless `pseudos`,
`ecutwfc`, and `ecutrho` are supplied through overrides.
