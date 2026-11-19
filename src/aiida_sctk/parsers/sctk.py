"""Parser for SCTK kel: the plain-text readers live in ``raw.py``; here their results become nodes."""
import numpy as np
from aiida import orm
from aiida.parsers import Parser

from .raw import parse_head_file, parse_kel_stdout  # noqa: F401  (re-exported)

HEAD_FILES = {  # retrieved file -> (output port, array name in ``kernel``)
    'aiida.pi0_head.dat': ('polarizability', 'pi0_head'),
    'aiida.ixc_pi0_head.dat': ('ixc_polarizability', 'ixc_pi0_head'),
    'aiida.chis_head.dat': ('spin_susceptibility', 'chis_head'),
}


def add_head_arrays(kernel, heads):
    """Store the parsed head files in ``kernel`` as complex arrays of shape (n_q, n_frequency).

    ``pi0_head`` and ``chis_head``: 1/eV per cell, both spins, negative (SCTK convention);
    ``ixc_pi0_head``: dimensionless.  Only q points present in ``kernel`` are kept, in its order.
    """
    indices = kernel.get_array('qpoint_indices').tolist()
    nfreq = len(kernel.get_array('frequencies'))
    for name, parsed in heads.items():
        rows = {}
        for block in parsed['qpoints']:
            if block.get('channel', 2) == 2 and len(block['values']) == nfreq:
                rows[block['index']] = [value[-1] for value in block['values']]
        if all(index in rows for index in indices):
            kernel.set_array(name, np.array([rows[index] for index in indices]))
    kernel.base.attributes.set('head_units', 'pi0_head, chis_head: 1/eV per cell, both spins, negative; '
                                             'ixc_pi0_head: dimensionless')


def build_kernel_array(parsed):
    """Collect the per-q kernels of ``parse_kel_stdout`` into an ``ArrayData`` (or ``None``)."""
    entries = [e for e in parsed.get('computed_qpoints', []) if 'mu_coulomb' in e]
    if not entries:
        return None
    frequencies = np.array([row[0] for row in entries[0]['mu_coulomb']])
    finite = np.isfinite(frequencies)
    kernel = orm.ArrayData()
    kernel.set_array('qpoint_indices', np.array([e['index'] for e in entries], dtype=int))
    kernel.set_array('frequencies', frequencies[finite])
    kernel.set_array('eig_max', np.array([e.get('eig_max', np.nan) for e in entries]))
    kernel.set_array('dos_ef', np.array([e.get('dos_ef', np.nan) for e in entries]))
    for key in ('mu_coulomb', 'mu_spin_fluctuation'):
        if not all(key in e and len(e[key]) == len(frequencies) for e in entries):
            continue
        values = np.array([[row[1] for row in e[key]] for e in entries])
        kernel.set_array(key, values[:, finite])
        if (~finite).any():
            kernel.set_array(f'{key}_infinite_frequency', values[:, ~finite][:, 0])
    if 'qpoints_cartesian' in parsed:
        kernel.set_array('qpoints_cartesian', np.array(parsed['qpoints_cartesian']))
    kernel.base.attributes.set_many({
        'frequency_units': 'Ry (imaginary axis)', 'dos_units': '1/Ry',
        'kernel_definition': 'N(0) <K(q, i nu)>_FS, dimensionless',
        'qpoint_units': '2 pi / alat (Cartesian)', 'q_grid': parsed.get('q_grid'),
        'q_shift': parsed.get('q_shift'), 'number_of_qpoints': parsed.get('number_of_qpoints'),
    })
    return kernel


class SCTKParser(Parser):
    def parse(self, **kwargs):
        repository = self.retrieved.base.repository
        names = repository.list_object_names()
        stdout = self.node.get_option('output_filename')
        if stdout not in names:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
        text = repository.get_object_content(stdout)
        if '%%%%%%%%' in text or 'Error in routine' in text:
            return self.exit_codes.ERROR_SCTK_FAILED
        if 'JOB DONE.' not in text:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        if 'aiida.pi0_head.dat' not in names:
            return self.exit_codes.ERROR_POLARIZABILITY_MISSING
        heads = {}
        for name, port in [(key, value[0]) for key, value in HEAD_FILES.items()] + \
                [('aiida.pi0_block.dat', 'polarizability_block')]:
            if name in names:
                with repository.open(name, 'rb') as handle:
                    self.out(port, orm.SinglefileData(file=handle, filename=name))
                if name in HEAD_FILES:
                    heads[HEAD_FILES[name][1]] = parse_head_file(repository.get_object_content(name))
        parsed = parse_kel_stdout(text)
        kernel = build_kernel_array(parsed)
        if kernel is not None:
            add_head_arrays(kernel, heads)
            self.out('kernel', kernel)
        summary = {key: value for key, value in parsed.items() if key != 'computed_qpoints'}
        summary['computed_qpoint_indices'] = [e['index'] for e in parsed['computed_qpoints']]
        summary['eig_max'] = [e.get('eig_max') for e in parsed['computed_qpoints']]
        summary['ixc_pi0_head'] = [e.get('ixc_pi0_head') for e in parsed['computed_qpoints']]
        summary['dos_ef'] = next((e['dos_ef'] for e in parsed['computed_qpoints'] if 'dos_ef' in e), None)
        summary = _clean(summary)
        summary.update({'calculation': 'kel', 'completed': True,
                        'response_files': [n for n in names if n.endswith('_head.dat')]})
        self.out('output_parameters', orm.Dict(dict=summary))


def _clean(value):
    """Replace NaN / infinity (not valid JSON) by None, recursively."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value
