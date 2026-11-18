from io import BytesIO
import pytest
from aiida import orm
from aiida.common.folders import SandboxFolder
from aiida.common.exceptions import InputValidationError
from aiida.engine.utils import instantiate_process
from aiida.plugins import CalculationFactory, WorkflowFactory, ParserFactory
from aiida_sctk.workflows.functions import twin_points, generate_twin_kpoints


def test_twin_order():
    points = twin_points([4,4,4], [0,0,0])
    assert len(points) == 128
    assert points[:64] == points[64:]
    assert points[:4] == [[0,0,0], [0,0,.25], [0,0,-.5], [0,0,-.25]]
    assert twin_points([2,1,1], [1,0,0])[2:] == [[.25,0,0],[-.25,0,0]]
    with pytest.raises(ValueError):
        twin_points([0,4,4], [0,0,0])


def test_entrypoints():
    assert CalculationFactory('sctk.kel').spec().inputs['dense_parent_folder'].required
    assert ParserFactory('sctk.kel')
    base_spec = WorkflowFactory('sctk.base').spec()
    assert base_spec.inputs['sctk']['dense_parent_folder'].required
    assert 'max_iterations' in base_spec.inputs
    assert 'polarizability' in base_spec.outputs
    spec = WorkflowFactory('sctk.polarizability').spec()
    assert 'kpoints' not in spec.inputs['twin']
    assert 'structure' not in spec.inputs['scf']['pw']
    assert 'parent_folder' not in spec.inputs['sctk']['sctk']
    assert spec.inputs['twin_mesh'].valid_type is orm.KpointsData
    assert {name: spec.exit_codes[name].status for name in (
        'ERROR_SCF', 'ERROR_NSCF', 'ERROR_DOS', 'ERROR_TWIN', 'ERROR_SCTK')} == {
            'ERROR_SCF': 401, 'ERROR_NSCF': 402, 'ERROR_DOS': 403,
            'ERROR_TWIN': 404, 'ERROR_SCTK': 405}


def test_protocols():
    from aiida_sctk.workflows.base import SCTKBaseWorkChain
    from aiida_sctk.workflows.polarizability import PolarizabilityWorkChain

    for workflow in (SCTKBaseWorkChain, PolarizabilityWorkChain):
        assert workflow.get_default_protocol() == 'balanced'
        assert set(workflow.get_available_protocols()) == {'fast', 'balanced', 'stringent'}

    base_fast = SCTKBaseWorkChain.get_protocol_inputs('fast')
    base_balanced = SCTKBaseWorkChain.get_protocol_inputs('balanced')
    base_stringent = SCTKBaseWorkChain.get_protocol_inputs('stringent')
    assert base_fast['sctk']['parameters']['KEL']['nq1'] == 2
    assert base_balanced['sctk']['parameters']['KEL']['nq1'] == 4
    assert base_stringent['sctk']['parameters']['KEL']['nq1'] == 6

    polar = PolarizabilityWorkChain.get_protocol_inputs(
        'balanced', overrides={'twin_mesh': {'mesh': [8, 8, 8]}}
    )
    assert polar['twin_mesh'] == {'mesh': [8, 8, 8], 'offset': [0.0, 0.0, 0.0]}
    assert polar['dos']['parameters']['DOS']['deltae'] == 0.01


def test_base_builder_from_protocol(aiida_profile, aiida_localhost):
    from aiida_sctk.workflows.base import SCTKBaseWorkChain

    code = orm.InstalledCode(
        computer=aiida_localhost, filepath_executable='/bin/true',
        default_calc_job_plugin='sctk.kel',
    ).store()
    parent = orm.RemoteData(computer=aiida_localhost, remote_path='/tmp/twin')
    dense = orm.RemoteData(computer=aiida_localhost, remote_path='/tmp/dense')
    builder = SCTKBaseWorkChain.get_builder_from_protocol(
        code, parent, dense, protocol='fast', options={'max_wallclock_seconds': 60}
    )
    assert builder.sctk.parent_folder is parent
    assert builder.sctk.dense_parent_folder is dense
    assert builder.sctk.parameters.get_dict()['KEL']['nq1'] == 2
    assert builder.sctk.metadata.options.max_wallclock_seconds == 60
    assert builder.max_iterations.value == 5


def test_provenance(aiida_profile):
    mesh = orm.KpointsData()
    mesh.set_kpoints_mesh([2,2,2], offset=[0.,0.,0.])
    points = generate_twin_kpoints(mesh)
    assert points.is_stored
    assert len(points.get_kpoints()) == 16
    assert points.creator.process_type


@pytest.fixture
def calc(aiida_profile, aiida_localhost, aiida_manager):
    computer = aiida_localhost
    code = orm.InstalledCode(computer=computer, filepath_executable='/bin/true',
                             default_calc_job_plugin='sctk.kel').store()
    inputs = dict(code=code, parent_folder=orm.RemoteData(computer=computer, remote_path='/tmp/twin'),
        dense_parent_folder=orm.RemoteData(computer=computer, remote_path='/tmp/dense'),
        parameters=orm.Dict(dict={'CONTROL': {'lshift_q':[0,0,0]}, 'KEL': {'nq1':4, 'nci':5}}),
        metadata={'options':{'resources':{'num_machines':1,'num_mpiprocs_per_machine':1}}})
    return instantiate_process(aiida_manager.get_runner(), CalculationFactory('sctk.kel'), **inputs)


def test_submission(calc):
    with SandboxFolder() as folder:
        info = calc.prepare_for_submission(folder)
        with folder.open('aiida.in') as handle:
            text = handle.read()
    assert "prefix = 'aiida'" in text
    assert "calculation = 'kel'" in text
    assert 'lshift_q(1)' in text
    assert info.remote_copy_list[-1][1:] == ('/tmp/dense/out/aiida.a2Fsave','out/aiida.a2Fsave')
    assert not info.remote_symlink_list
    assert 'aiida.pi0_head.dat' in info.retrieve_list


def test_blocked_prefix():
    with pytest.raises(InputValidationError):
        CalculationFactory('sctk.kel').set_blocked_keywords({'CONTROL': {'prefix':'wrong'}})


@pytest.mark.parametrize('stdout,files,error', [
    ('JOB DONE.', ['aiida.pi0_head.dat'], None),
    ('JOB DONE.', [], 'ERROR_POLARIZABILITY_MISSING'),
    ('Error in routine', [], 'ERROR_SCTK_FAILED'),
    ('unfinished', [], 'ERROR_OUTPUT_STDOUT_INCOMPLETE'),
])
def test_parser(aiida_profile, stdout, files, error):
    from types import SimpleNamespace
    parser = ParserFactory('sctk.kel')
    outputs = {}
    repository = SimpleNamespace(
        list_object_names=lambda: ['aiida.out']+files,
        get_object_content=lambda _:stdout,
        open=lambda *args:BytesIO(b'# raw response\n1 2 3\n'))
    errors = {name:name for name in ('ERROR_OUTPUT_STDOUT_MISSING',
        'ERROR_OUTPUT_STDOUT_INCOMPLETE', 'ERROR_POLARIZABILITY_MISSING', 'ERROR_SCTK_FAILED')}
    instance = SimpleNamespace(retrieved=SimpleNamespace(base=SimpleNamespace(repository=repository)),
        node=SimpleNamespace(get_option=lambda _:'aiida.out'),
        exit_codes=SimpleNamespace(**errors), out=lambda k,v:outputs.update({k:v}))
    assert parser.parse(instance) == error
    if error is None:
        assert outputs['polarizability'].get_content() == '# raw response\n1 2 3\n'


@pytest.mark.parametrize('stage,calculation', [('nscf','nscf'), ('twin','bands')])
def test_pw_wiring(aiida_profile, stage, calculation):
    from types import SimpleNamespace
    from aiida_sctk.workflows.polarizability import PolarizabilityWorkChain
    captured = {}
    parameters = orm.Dict(dict={'SYSTEM':{'nbnd':20}, 'CONTROL':{}})
    parent = object()
    points = object()
    structure = object()
    instance = SimpleNamespace(
        inputs=SimpleNamespace(structure=structure),
        ctx=SimpleNamespace(twin_kpoints=points),
        exposed_inputs=lambda *args:{'pw':{'parameters':parameters}},
        submit=lambda process,**kwargs:captured.update(kwargs))
    PolarizabilityWorkChain.run_pw(instance, stage, calculation, parent)
    assert captured['pw']['parent_folder'] is parent
    assert captured['pw']['structure'] is structure
    assert captured['clean_workdir'].value is False
    assert captured['pw']['settings'].get_dict()['PARENT_FOLDER_SYMLINK'] is False
    params = captured['pw']['parameters'].get_dict()
    assert params['CONTROL']['calculation'] == calculation
    if stage == 'nscf':
        assert params['SYSTEM']['la2f'] is True
        assert params['SYSTEM']['occupations'] == 'tetrahedra_opt'
    else:
        assert captured['kpoints'] is points
        assert params['SYSTEM']['la2f'] is False
    assert parameters.get_dict() == {'SYSTEM':{'nbnd':20}, 'CONTROL':{}}


def _kel_text():
    from pathlib import Path
    return (Path(__file__).parent / 'files' / 'kel_V_q4_shifted.out').read_text()


def test_parse_kel_stdout():
    """Stoner eigenvalues and averaged kernels of a real bcc V run (4x4x4 shifted q grid)."""
    import numpy as np
    from aiida_sctk.parsers.sctk import build_kernel_array, parse_kel_stdout

    parsed = parse_kel_stdout(_kel_text())
    assert parsed['q_grid'] == [4, 4, 4] and parsed['q_shift'] == [1, 1, 1]
    assert parsed['number_of_qpoints'] == 6 and len(parsed['qpoints_cartesian']) == 6
    assert [e['index'] for e in parsed['computed_qpoints']] == [1, 2, 3, 4, 5, 6]
    assert [e['eig_max'] for e in parsed['computed_qpoints']] == pytest.approx(
        [0.63725, 0.70330, 0.68400, 0.77160, 0.77382, 0.73520])
    kernel = build_kernel_array(parsed)
    assert kernel.get_array('frequencies').shape == (8,)
    assert np.isfinite(kernel.get_array('frequencies')).all()
    assert kernel.get_array('mu_spin_fluctuation').shape == (6, 8)
    assert kernel.get_array('mu_spin_fluctuation')[0, 0] == pytest.approx(0.8865)
    assert kernel.get_array('mu_coulomb')[0, 1] == pytest.approx(0.9418)
    assert kernel.get_array('mu_coulomb_infinite_frequency')[0] == pytest.approx(15.71)


def test_irreducible_qpoints_and_split():
    """spglib counts agree with what sctk.x reported on LUMI for the same grids."""
    from ase.build import bulk
    from ase import Atoms
    from aiida_sctk.workflows.functions import count_irreducible_qpoints, split_qpoints

    vanadium = orm.StructureData(ase=bulk('V', 'bcc', a=2.99723))
    assert count_irreducible_qpoints(vanadium, [4, 4, 4], [1, 1, 1]) == 6
    a = 6.3226
    half_heusler = orm.StructureData(ase=Atoms(
        'HfSbRu', cell=[[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]], pbc=True,
        positions=[[0, 0, 0], [a/2, 0, 0], [a/4, a/4, a/4]]))
    assert count_irreducible_qpoints(half_heusler, [6, 6, 6], [0, 0, 0]) == 16
    assert split_qpoints(2, 16, 4) == [[2, 5], [6, 9], [10, 13], [14, 16]]
    assert split_qpoints(1, 6, 10) == [[1, 6]]
    with pytest.raises(ValueError):
        split_qpoints(3, 2, 1)


def test_merge_kel_kernels():
    import numpy as np
    from aiida_sctk.parsers.sctk import build_kernel_array, parse_kel_stdout
    from aiida_sctk.workflows.functions import merge_kel_kernels

    parsed = parse_kel_stdout(_kel_text())
    first, second = dict(parsed), dict(parsed)
    first['computed_qpoints'] = parsed['computed_qpoints'][3:]
    second['computed_qpoints'] = parsed['computed_qpoints'][:3]
    merged = merge_kel_kernels(a=build_kernel_array(first), b=build_kernel_array(second))
    full = build_kernel_array(parsed)
    assert merged.get_array('qpoint_indices').tolist() == [1, 2, 3, 4, 5, 6]
    for name in ('eig_max', 'mu_coulomb', 'mu_spin_fluctuation', 'mu_coulomb_infinite_frequency'):
        assert np.allclose(merged.get_array(name), full.get_array(name))
    with pytest.raises(Exception):
        merge_kel_kernels(a=build_kernel_array(first), b=build_kernel_array(first))


def test_spin_fluctuation_entry_point():
    spec = WorkflowFactory('sctk.spin_fluctuation').spec()
    assert 'qpoints_per_job' in spec.inputs and 'q_ranges' in spec.inputs
    assert 'kernel' in spec.outputs and 'sctk' not in spec.outputs
    assert spec.exit_codes.ERROR_QPOINT_COVERAGE.status == 406
    calc_spec = CalculationFactory('sctk.kel').spec()
    assert 'kernel' in calc_spec.outputs and 'spin_susceptibility' in calc_spec.outputs
