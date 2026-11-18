"""SCF → dense NSCF → DOS → twin bands → SCTK kel."""
from copy import deepcopy
from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida_sctk.workflows.base import SCTKBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from .functions import generate_twin_kpoints

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosCalculation = CalculationFactory('quantumespresso.dos')


def validate_twin_mesh(value, _):
    """Validate that the k-points data contains a mesh with an SCTK half-grid offset."""
    try:
        _, offset = value.get_kpoints_mesh()
    except (AttributeError, ValueError):
        return 'twin_mesh must define a k-point mesh, not an explicit point list'
    if any(abs(component * 2 - round(component * 2)) > 1.e-12
           or round(component * 2) not in (0, 1) for component in offset):
        return 'twin_mesh offset components must be either 0 or 0.5'
    return None


class PolarizabilityWorkChain(ProtocolMixin, WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('twin_mesh', valid_type=orm.KpointsData, validator=validate_twin_mesh,
                   help='Uniform mesh and offset used to construct the doubled SCTK twin grid.')
        for stage in ('scf', 'nscf', 'twin'):
            excluded = ['pw.structure', 'pw.parent_folder', 'clean_workdir']
            if stage == 'twin':
                excluded += ['kpoints', 'kpoints_distance', 'kpoints_force_parity']
            spec.expose_inputs(PwBaseWorkChain, namespace=stage, exclude=excluded)
        spec.expose_inputs(DosCalculation, namespace='dos', exclude=['parent_folder'])
        spec.expose_inputs(SCTKBaseWorkChain, namespace='sctk',
                           exclude=['sctk.parent_folder', 'sctk.dense_parent_folder'])
        spec.expose_outputs(SCTKBaseWorkChain, namespace='sctk')
        spec.expose_outputs(DosCalculation, namespace='dos')
        spec.outline(cls.setup, cls.run_scf, cls.inspect_scf,
                     cls.run_nscf, cls.inspect_nscf, cls.run_dos, cls.inspect_dos,
                     cls.run_twin, cls.inspect_twin, cls.run_sctk, cls.results)
        spec.exit_code(401, 'ERROR_SCF', message='SCF step failed.')
        spec.exit_code(402, 'ERROR_NSCF', message='NSCF step failed.')
        spec.exit_code(403, 'ERROR_DOS', message='DOS step failed.')
        spec.exit_code(404, 'ERROR_TWIN', message='Twin-grid bands step failed.')
        spec.exit_code(405, 'ERROR_SCTK', message='SCTK step failed.')

    @classmethod
    def get_protocol_filepath(cls):
        """Return the path to the protocol configuration file."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'polarizability.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, dos_code, sctk_code, structure, protocol=None, overrides=None,
        options=None, **kwargs,
    ):
        """Return a builder populated for the complete polarizability workflow."""
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(dos_code, str):
            dos_code = orm.load_code(dos_code)
        if isinstance(sctk_code, str):
            sctk_code = orm.load_code(sctk_code)

        inputs = cls.get_protocol_inputs(protocol, overrides)
        pw_builders = {}
        for stage in ('scf', 'nscf', 'twin'):
            pw_builder = PwBaseWorkChain.get_builder_from_protocol(
                pw_code, structure, protocol, overrides=inputs.get(stage),
                options=options, **kwargs,
            )
            pw_builder.pw.pop('structure', None)
            pw_builder.pop('clean_workdir', None)
            if stage == 'nscf':
                system = pw_builder.pw.parameters.get_dict()['SYSTEM']
                system.pop('smearing', None)
                system.pop('degauss', None)
                parameters = pw_builder.pw.parameters.get_dict()
                parameters['SYSTEM'] = system
                pw_builder.pw.parameters = orm.Dict(dict=parameters)
            elif stage == 'twin':
                pw_builder.pop('kpoints', None)
                pw_builder.pop('kpoints_distance', None)
                pw_builder.pop('kpoints_force_parity', None)
                parameters = pw_builder.pw.parameters.get_dict()
                for key in ('occupations', 'smearing', 'degauss'):
                    parameters['SYSTEM'].pop(key, None)
                pw_builder.pw.parameters = orm.Dict(dict=parameters)
            pw_builders[stage] = pw_builder

        metadata_dos = inputs['dos']['metadata']
        if options:
            metadata_dos['options'] = recursive_merge(metadata_dos['options'], options)
        metadata_dos['options'] = cls.set_default_resources(
            metadata_dos['options'], dos_code.computer.scheduler_type
        )

        sctk = SCTKBaseWorkChain.get_builder_from_protocol(
            sctk_code, protocol=protocol, overrides=inputs.get('sctk'), options=options
        )
        twin_spec = inputs['twin_mesh']
        twin_mesh = orm.KpointsData()
        twin_mesh.set_kpoints_mesh(twin_spec['mesh'], offset=twin_spec['offset'])

        builder = cls.get_builder()
        builder.structure = structure
        builder.twin_mesh = twin_mesh
        builder.scf = pw_builders['scf']
        builder.nscf = pw_builders['nscf']
        builder.twin = pw_builders['twin']
        builder.dos.code = dos_code
        builder.dos.parameters = orm.Dict(dict=inputs['dos']['parameters'])
        builder.dos.metadata = metadata_dos
        if 'settings' in inputs['dos']:
            builder.dos.settings = orm.Dict(dict=inputs['dos']['settings'])
        builder.sctk = sctk
        return builder

    def setup(self):
        self.ctx.twin_kpoints = generate_twin_kpoints(self.inputs.twin_mesh)

    def run_pw(self, stage, calculation, parent=None):
        inputs = self.exposed_inputs(PwBaseWorkChain, stage)
        inputs['pw']['structure'] = self.inputs.structure
        params = deepcopy(inputs['pw']['parameters'].get_dict())
        params.setdefault('CONTROL', {})['calculation'] = calculation
        if stage == 'nscf':
            params.setdefault('SYSTEM', {}).update(la2f=True, occupations='tetrahedra_opt')
        if stage == 'twin':
            params.setdefault('SYSTEM', {})['la2f'] = False
            # Bands preserves the explicit double-grid ordering used by SCTK.
            inputs['kpoints'] = self.ctx.twin_kpoints
        settings = inputs['pw']['settings'].get_dict() if 'settings' in inputs['pw'] else {}
        settings['PARENT_FOLDER_SYMLINK'] = False
        inputs['pw']['settings'] = orm.Dict(dict=settings)
        inputs['pw']['parameters'] = orm.Dict(dict=params)
        inputs['clean_workdir'] = orm.Bool(False)
        if parent is not None:
            inputs['pw']['parent_folder'] = parent
        return ToContext(**{stage: self.submit(PwBaseWorkChain, **inputs)})

    def run_scf(self):
        return self.run_pw('scf', 'scf')

    def inspect_scf(self):
        if not self.ctx.scf.is_finished_ok:
            return self.exit_codes.ERROR_SCF

    def run_nscf(self):
        return self.run_pw('nscf', 'nscf', self.ctx.scf.outputs.remote_folder)

    def inspect_nscf(self):
        if not self.ctx.nscf.is_finished_ok:
            return self.exit_codes.ERROR_NSCF

    def run_dos(self):
        inputs = self.exposed_inputs(DosCalculation, 'dos')
        inputs['parent_folder'] = self.ctx.nscf.outputs.remote_folder
        return ToContext(dos=self.submit(DosCalculation, **inputs))

    def inspect_dos(self):
        if not self.ctx.dos.is_finished_ok:
            return self.exit_codes.ERROR_DOS

    def run_twin(self):
        return self.run_pw('twin', 'bands', self.ctx.scf.outputs.remote_folder)

    def inspect_twin(self):
        if not self.ctx.twin.is_finished_ok:
            return self.exit_codes.ERROR_TWIN

    def run_sctk(self):
        inputs = self.exposed_inputs(SCTKBaseWorkChain, 'sctk')
        inputs['sctk']['parent_folder'] = self.ctx.twin.outputs.remote_folder
        inputs['sctk']['dense_parent_folder'] = self.ctx.nscf.outputs.remote_folder
        params = deepcopy(inputs['sctk']['parameters'].get_dict())
        _, offset = self.inputs.twin_mesh.get_kpoints_mesh()
        params.setdefault('CONTROL', {})['lshift_q'] = [round(value * 2) for value in offset]
        inputs['sctk']['parameters'] = orm.Dict(dict=params)
        return ToContext(sctk=self.submit(SCTKBaseWorkChain, **inputs))

    def results(self):
        if not self.ctx.sctk.is_finished_ok:
            return self.exit_codes.ERROR_SCTK
        self.out_many(self.exposed_outputs(self.ctx.sctk, SCTKBaseWorkChain, namespace='sctk'))
        self.out_many(self.exposed_outputs(self.ctx.dos, DosCalculation, namespace='dos'))
