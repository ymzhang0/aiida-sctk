"""Base work chain for SCTK calculations."""

from aiida.common import AttributeDict
from aiida.engine import BaseRestartWorkChain, while_

from aiida_sctk.calculations.sctk import SCTKCalculation
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin


class SCTKBaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Run an SCTK calculation with the standard restart-work-chain machinery."""

    _process_class = SCTKCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(SCTKCalculation, namespace='sctk')
        spec.expose_outputs(SCTKCalculation)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.exit_code(
            300,
            'ERROR_UNRECOVERABLE_FAILURE',
            message='The SCTK calculation failed with an unrecoverable error.',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return the path to the protocol configuration file."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'base.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, parent_folder=None, dense_parent_folder=None, protocol=None,
        overrides=None, options=None,
    ):
        """Return a builder populated from an SCTK protocol."""
        from aiida import orm
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(code, str):
            code = orm.load_code(code)

        inputs = cls.get_protocol_inputs(protocol, overrides)
        metadata = inputs['sctk']['metadata']
        if options:
            metadata['options'] = recursive_merge(metadata['options'], options)
        metadata['options'] = cls.set_default_resources(
            metadata['options'], code.computer.scheduler_type
        )

        builder = cls.get_builder()
        builder.sctk.code = code
        builder.sctk.parameters = orm.Dict(dict=inputs['sctk']['parameters'])
        builder.sctk.metadata = metadata
        if 'settings' in inputs['sctk']:
            builder.sctk.settings = orm.Dict(dict=inputs['sctk']['settings'])
        if parent_folder is not None:
            builder.sctk.parent_folder = parent_folder
        if dense_parent_folder is not None:
            builder.sctk.dense_parent_folder = dense_parent_folder
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.max_iterations = orm.Int(inputs['max_iterations'])
        return builder

    def setup(self):
        """Initialize the restart loop and inputs for the SCTK calculation."""
        super().setup()
        self.ctx.restart_calc = None
        self.ctx.inputs = AttributeDict(self.exposed_inputs(SCTKCalculation, 'sctk'))

    def report_error_handled(self, calculation, action):
        """Report a failed calculation and the action taken by an error handler."""
        arguments = (
            calculation.process_label,
            calculation.pk,
            calculation.exit_status,
            calculation.exit_message,
        )
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report(f'Action taken: {action}')
