"""SCTK kel calculation using the patched QE/SCTK build."""
from pathlib import PurePosixPath
from aiida import orm
from aiida.common.exceptions import InputValidationError
from aiida_quantumespresso.calculations.namelists import NamelistsCalculation


class SCTKCalculation(NamelistsCalculation):
    """Copy twin output and dense-grid a2Fsave into an isolated kel directory."""
    _default_namelists = ['CONTROL', 'KEL']
    _default_parser = 'sctk.kel'
    _blocked_keywords = [('CONTROL', 'prefix', 'aiida'),
                         ('CONTROL', 'outdir', './out/'),
                         ('CONTROL', 'calculation', 'kel'),
                         ('CONTROL', 'restart_mode', 'from_scratch')]
    _internal_retrieve_list = ['aiida.pi0_head.dat', 'aiida.ixc_pi0_head.dat']

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('parent_folder', valid_type=orm.RemoteData, required=True,
                   help='Twin bands PW remote folder, with out/aiida.save and wavefunctions.')
        spec.input('dense_parent_folder', valid_type=orm.RemoteData,
                   help='Dense NSCF folder containing out/aiida.a2Fsave (SYSTEM.la2f=true).')
        spec.input('parameters', valid_type=orm.Dict, required=True)
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output('polarizability', valid_type=orm.SinglefileData)
        spec.output('ixc_polarizability', valid_type=orm.SinglefileData, required=False)
        spec.exit_code(320, 'ERROR_SCTK_FAILED', message='SCTK reported a fatal error.')
        spec.exit_code(321, 'ERROR_POLARIZABILITY_MISSING', message='No pi0 head output was produced.')

    def prepare_for_submission(self, folder):
        for name in ('parent_folder', 'dense_parent_folder'):
            if self.inputs[name].computer.uuid != self.inputs.code.computer.uuid:
                raise InputValidationError(f'{name} and code must use the same computer')
        settings = self.inputs.settings.get_dict() if 'settings' in self.inputs else {}
        forbidden = {'PARENT_FOLDER_SYMLINK', 'PARENT_CALC_OUT_SUBFOLDER', 'NAMELISTS'}
        if forbidden.intersection(k.upper() for k in settings):
            raise InputValidationError('SCTK requires private copies of the standard out/ folder')
        parameters = self.inputs.parameters.get_dict()
        if set(parameters) - {'CONTROL', 'KEL'}:
            raise InputValidationError('Only CONTROL and KEL namelists are supported')
        info = super().prepare_for_submission(folder)
        dense = self.inputs.dense_parent_folder
        info.remote_copy_list.append((dense.computer.uuid,
            str(PurePosixPath(dense.get_remote_path()) / 'out/aiida.a2Fsave'),
            'out/aiida.a2Fsave'))
        return info
