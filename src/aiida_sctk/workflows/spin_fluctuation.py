"""SCF -> dense NSCF -> DOS -> twin bands -> SCTK kel split over the irreducible q points."""
from copy import deepcopy

from aiida import orm
from aiida.engine import append_

from aiida_sctk.workflows.base import SCTKBaseWorkChain
from aiida_sctk.workflows.polarizability import DosCalculation, PolarizabilityWorkChain

from .functions import count_irreducible_qpoints, merge_kel_kernels, split_qpoints


def validate_q_ranges(value, _):
    """Validate a list of inclusive ``[start_q, last_q]`` pairs."""
    if value is None:
        return None
    ranges = value.get_list()
    if not ranges or any(len(r) != 2 or type(r[0]) is not int or type(r[1]) is not int
                         or r[0] < 1 or r[1] < r[0] for r in ranges):
        return 'q_ranges must be a non-empty list of [start_q, last_q] pairs with 1 <= start_q <= last_q'
    return None


class SpinFluctuationWorkChain(PolarizabilityWorkChain):
    """Imaginary-axis Coulomb / spin-fluctuation kernel of SCTK for every irreducible q point.

    The PW steps are run once.  The kel step is submitted as several independent jobs, each
    covering a range ``KEL.start_q .. KEL.last_q``, and their per-q Stoner eigenvalues and
    Fermi-surface averaged kernels are merged into a single ``kernel`` output.  ``KEL.lsf`` is
    forced to 1.  The q-point average (which needs the star weights) and the conversion to an
    EPW ``fila2f_sf`` table are deliberately left to post-processing.
    """

    @classmethod
    def define(cls, spec):
        # skip PolarizabilityWorkChain.define: the single-kel outputs do not apply here
        super(PolarizabilityWorkChain, cls).define(spec)
        cls.define_common(spec)
        spec.input('qpoints_per_job', valid_type=orm.Int, required=False,
                   help='Split the irreducible q points over kel jobs of at most this many q points. '
                        'If neither this nor `q_ranges` is given a single job computes all of them.')
        spec.input('q_ranges', valid_type=orm.List, required=False, validator=validate_q_ranges,
                   help='Explicit inclusive [start_q, last_q] ranges, one kel job each (overrides `qpoints_per_job`).')
        spec.input('skip_gamma', valid_type=orm.Bool, default=lambda: orm.Bool(True),
                   help='Skip q = Gamma (the first q point of an unshifted grid).')
        spec.output('kernel', valid_type=orm.ArrayData,
                    help='Merged per-q arrays: qpoint_indices, frequencies [Ry], eig_max, dos_ef, '
                         'mu_coulomb and mu_spin_fluctuation (q, frequency).')
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.outline(cls.setup, cls.run_scf, cls.inspect_scf,
                     cls.run_nscf, cls.inspect_nscf, cls.run_dos, cls.inspect_dos,
                     cls.run_twin, cls.inspect_twin, cls.run_sctk, cls.results)
        spec.exit_code(406, 'ERROR_QPOINT_COVERAGE',
                       message='The kel jobs did not cover the irreducible q points exactly once.')
        spec.exit_code(407, 'ERROR_MISSING_Q_GRID',
                       message='KEL.nq1, nq2 and nq3 must be given in sctk.sctk.parameters.')

    def get_shift(self):
        _, offset = self.inputs.twin_mesh.get_kpoints_mesh()
        return [round(value * 2) for value in offset]

    def get_q_ranges(self, kel):
        """Return the list of [start_q, last_q] (last_q = 0: up to the last one) and the expected count."""
        shift = self.get_shift()
        first = 2 if (self.inputs.skip_gamma.value and not any(shift)) else 1
        if 'q_ranges' in self.inputs:
            return self.inputs.q_ranges.get_list(), None, first
        if 'qpoints_per_job' not in self.inputs:
            return [[first, 0]], None, first
        mesh = [kel['nq1'], kel['nq2'], kel['nq3']]
        number = count_irreducible_qpoints(self.inputs.structure, mesh, shift)
        return split_qpoints(first, number, self.inputs.qpoints_per_job.value), number, first

    def run_sctk(self):
        base = self.exposed_inputs(SCTKBaseWorkChain, 'sctk')
        params = deepcopy(base['sctk']['parameters'].get_dict())
        kel = params.setdefault('KEL', {})
        if not all(key in kel for key in ('nq1', 'nq2', 'nq3')):
            return self.exit_codes.ERROR_MISSING_Q_GRID
        kel['lsf'] = 1
        params.setdefault('CONTROL', {})['lshift_q'] = self.get_shift()
        ranges, expected, first = self.get_q_ranges(kel)
        self.ctx.q_ranges, self.ctx.expected_qpoints, self.ctx.first_q = ranges, expected, first
        self.report(f'submitting {len(ranges)} kel job(s) for q ranges {ranges}')
        for start, last in ranges:
            inputs = deepcopy(base)
            job_params = deepcopy(params)
            job_params['KEL'].update(start_q=start, last_q=last)
            inputs['sctk']['parameters'] = orm.Dict(dict=job_params)
            inputs['sctk']['parent_folder'] = self.ctx.twin.outputs.remote_folder
            inputs['sctk']['dense_parent_folder'] = self.ctx.nscf.outputs.remote_folder
            inputs.setdefault('metadata', {})['call_link_label'] = f'kel_q{start}_{last}'
            self.to_context(kel=append_(self.submit(SCTKBaseWorkChain, **inputs)))

    def results(self):
        failed = [node.pk for node in self.ctx.kel if not node.is_finished_ok]
        if failed:
            self.report(f'kel work chain(s) {failed} failed')
            return self.exit_codes.ERROR_SCTK
        kernels, eig_max, ixc_head, indices, number = {}, {}, {}, [], None
        for node in self.ctx.kel:
            if 'kernel' not in node.outputs:
                self.report(f'kel work chain {node.pk} produced no kernel table')
                return self.exit_codes.ERROR_SCTK
            parameters = node.outputs.output_parameters.get_dict()
            number = parameters.get('number_of_qpoints', number)
            indices += parameters['computed_qpoint_indices']
            eig_max.update(zip(parameters['computed_qpoint_indices'], parameters['eig_max']))
            ixc_head.update(zip(parameters['computed_qpoint_indices'], parameters.get('ixc_pi0_head', [])))
            kernels[f'kel_{node.pk}'] = node.outputs.kernel
        expected = list(range(self.ctx.first_q, (number or 0) + 1))
        if sorted(indices) != expected or (self.ctx.expected_qpoints not in (None, number)):
            self.report(f'q-point coverage mismatch: computed {sorted(indices)}, sctk.x reports {number} '
                        f'irreducible q points, splitting assumed {self.ctx.expected_qpoints}')
            return self.exit_codes.ERROR_QPOINT_COVERAGE
        kernel = merge_kel_kernels(**kernels) if len(kernels) > 1 else next(iter(kernels.values()))
        self.out('kernel', kernel)
        reference = self.ctx.kel[0].outputs.output_parameters.get_dict()
        summary = {key: reference.get(key) for key in (
            'q_grid', 'q_shift', 'number_of_qpoints', 'qpoints_cartesian', 'frequencies',
            'dense_k_grid', 'number_of_bands', 'ecutfock', 'number_of_symmetries', 'dos_ef')}
        summary['computed_qpoint_indices'] = sorted(indices)
        summary['eig_max'] = [eig_max[index] for index in sorted(indices)]
        summary['ixc_pi0_head'] = [ixc_head.get(index) for index in sorted(indices)]
        summary['q_ranges'] = self.ctx.q_ranges
        self.out('output_parameters', orm.Dict(dict=summary).store())
        self.out_many(self.exposed_outputs(self.ctx.dos, DosCalculation, namespace='dos'))
