"""Run with `verdi run examples/builder.py ...`; constructs but never submits."""
import argparse
import json
from aiida import orm
from aiida.plugins import WorkflowFactory


def build(structure, pseudos, pw_code, dos_code, sctk_code):
    builder = WorkflowFactory('sctk.polarizability').get_builder()
    builder.structure = structure
    twin_mesh = orm.KpointsData()
    twin_mesh.set_kpoints_mesh([4,4,4], offset=[0.,0.,0.])
    builder.twin_mesh = twin_mesh
    module_text = ('module --force purge\nmodule load LUMI/25.03 partition/C '
                   'cpeGNU/25.03 cray-fftw cray-libsci')
    def options(ranks, threads):
        return dict(resources={'num_machines':1, 'num_mpiprocs_per_machine':ranks,
                               'num_cores_per_mpiproc':threads},
                    max_wallclock_seconds=1800, max_memory_kb=200*1024**2,
                    account='project_465002926', queue_name='debug', withmpi=True,
                    environment_variables={'OMP_NUM_THREADS':str(threads)},
                    prepend_text=module_text)
    pw_parameters = {
        'scf': {
            'CONTROL': {'calculation': 'scf'},
            'SYSTEM': {'ecutwfc': 84., 'ecutrho': 336., 'occupations': 'smearing',
                       'smearing': 'cold', 'degauss': 0.02},
            'ELECTRONS': {'conv_thr': 1.e-12, 'mixing_beta': 0.4},
        },
        'nscf': {
            'CONTROL': {'calculation': 'nscf'},
            'SYSTEM': {'ecutwfc': 84., 'ecutrho': 336., 'nbnd': 40},
            'ELECTRONS': {'diagonalization': 'cg'},
        },
        'twin': {
            'CONTROL': {'calculation': 'bands'},
            'SYSTEM': {'ecutwfc': 84., 'ecutrho': 336., 'nbnd': 40},
            'ELECTRONS': {'diagonalization': 'cg'},
        },
    }
    for name in ('scf', 'nscf', 'twin'):
        stage = builder[name]
        stage.pw.code = pw_code
        stage.pw.pseudos = pseudos
        stage.pw.parameters = orm.Dict(dict=pw_parameters[name])
        cmdline = ['-nk', '64'] + (['-nd', '1'] if name != 'scf' else [])
        stage.pw.settings = orm.Dict(dict={'CMDLINE': cmdline})
        stage.pw.metadata.options = options(128, 1)
    for name, dimensions in (('scf', [12,12,12]), ('nscf', [18,18,18])):
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh(dimensions)
        builder[name].kpoints = kpoints
    builder.dos.code = dos_code
    builder.dos.parameters = orm.Dict(dict={'DOS':{
        'emin':0., 'emax':30., 'deltae':.01, 'bz_sum':'tetrahedra_opt'}})
    builder.dos.metadata.options = options(1,1)
    builder.sctk.sctk.code = sctk_code
    builder.sctk.sctk.parameters = orm.Dict(dict={'CONTROL':{}, 'KEL':{
        'start_q':2, 'nci':5, 'laddxc':0, 'lsf':1, 'ecutfock':20.,
        'nq1':4, 'nq2':4, 'nq3':4}})
    builder.sctk.sctk.settings = orm.Dict(dict={'CMDLINE':['-nk','32']})
    builder.sctk.sctk.metadata.options = options(32,4)
    return builder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure', type=int, required=True)
    parser.add_argument('--pseudos', required=True, help='JSON kind-name to UpfData PK mapping')
    parser.add_argument('--pw', required=True)
    parser.add_argument('--dos', required=True)
    parser.add_argument('--sctk', required=True)
    args = parser.parse_args()
    builder = build(orm.load_node(args.structure),
        {k:orm.load_node(v) for k,v in json.loads(args.pseudos).items()},
        orm.load_code(args.pw), orm.load_code(args.dos), orm.load_code(args.sctk))
    error = builder._inputs(prune=True)
    validation = WorkflowFactory('sctk.polarizability').spec().inputs.validate(error)
    if validation:
        raise ValueError(validation)
    print(builder)
