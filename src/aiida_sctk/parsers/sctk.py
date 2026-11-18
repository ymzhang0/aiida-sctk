"""Parser for SCTK kel: response files are kept verbatim, the stdout summary is parsed."""
import re

import numpy as np
from aiida import orm
from aiida.parsers import Parser

_FLOAT = r'[-+]?(?:\d+\.?\d*(?:[EeDd][-+]?\d+)?|NaN|Infinity)'


def _to_float(token):
    token = token.replace('D', 'E').replace('d', 'e')
    if token.lower() == 'infinity':
        return float('inf')
    return float(token)


def _parse_table(block, title):
    """Return the two-column table that follows ``title`` (frequency, value) in ``block``."""
    match = re.search(re.escape(title) + r'\s*\n((?:\s+\S+\s+\S+\s*\n)+)', block)
    if match is None:
        return None
    return [[_to_float(x) for x in line.split()] for line in match.group(1).strip().splitlines()]


def parse_kel_stdout(text):
    """Parse the summary that ``sctk.x`` (calculation='kel') prints to stdout.

    :return: dictionary with the q grid, the list of irreducible q points (Cartesian, 2 pi / alat),
        the Chebyshev imaginary frequencies [Ry] and, for every computed q point, the Stoner
        diagnostics and the Fermi-surface averaged Coulomb / spin-fluctuation kernels.
    """
    result = {}
    match = re.search(r'q grid :\s*(\d+)\s+(\d+)\s+(\d+)', text)
    if match:
        result['q_grid'] = [int(x) for x in match.groups()]
    match = re.search(r'With a half shift\s+(\d+)\s+(\d+)\s+(\d+)', text)
    result['q_shift'] = [int(x) for x in match.groups()] if match else [0, 0, 0]
    for key, label in (('first_q', 'The first q'), ('last_q', 'The last q'), ('lsf', 'lsf'),
                       ('number_of_bands', 'Number of bands'), ('number_of_symmetries', 'Number of symmetries')):
        match = re.search(re.escape(label) + r' :\s*(-?\d+)', text)
        if match:
            result[key] = int(match.group(1))
    match = re.search(r'Dense k-grid :\s*(\d+)\s+(\d+)\s+(\d+)', text)
    if match:
        result['dense_k_grid'] = [int(x) for x in match.groups()]
    match = re.search(r'Cutoff kinetic energy :\s*(\S+)', text)
    if match:
        result['ecutfock'] = _to_float(match.group(1))

    match = re.search(r'\(\s*(\d+) q-points\):\s*\n\s*N\s+xq\(1\)\s+xq\(2\)\s+xq\(3\)\s*\n', text)
    if match:
        number = int(match.group(1))
        lines = text[match.end():].splitlines()[:number]
        result['number_of_qpoints'] = number
        result['qpoints_cartesian'] = [[float(x) for x in line.split()[1:4]] for line in lines]

    match = re.search(r'Index, Frequency \[Ry\]\s*\n((?:\s+\d+\s+\S+\s*\n)+)', text)
    if match:
        result['frequencies'] = [_to_float(line.split()[1]) for line in match.group(1).strip().splitlines()]

    computed = []
    pieces = re.split(r'\n\s*q-point :\s*(\d+)\s*\n', text)
    for index, block in zip(pieces[1::2], pieces[2::2]):
        entry = {'index': int(index)}
        for key, pattern in (
            ('ixc_pi0_head', r'\(I_\{xc\} Pi_0\)_\{00\} :\s*(\S+)'),
            ('eig_max', r'EigMax\(I_\{xc\} Pi_0\) :\s*(\S+)'),
            ('dos_ef', r'DOS\(EF\) =\s*(\S+)'),
            ('number_of_plane_waves', r'Number of PWs for W :\s*(\S+)'),
        ):
            found = re.search(pattern, block)
            if found:
                entry[key] = _to_float(found.group(1))
        for key, title in (('mu_coulomb', 'Coulomb parameter(mu)'), ('mu_spin_fluctuation', 'Spin-F. parameter(mu)')):
            table = _parse_table(block, title)
            if table is not None:
                entry[key] = table
        computed.append(entry)
    result['computed_qpoints'] = computed
    return result


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
        for name, port in [('aiida.pi0_head.dat', 'polarizability'),
                           ('aiida.ixc_pi0_head.dat', 'ixc_polarizability'),
                           ('aiida.chis_head.dat', 'spin_susceptibility'),
                           ('aiida.pi0_block.dat', 'polarizability_block')]:
            if name in names:
                with repository.open(name, 'rb') as handle:
                    self.out(port, orm.SinglefileData(file=handle, filename=name))
        parsed = parse_kel_stdout(text)
        kernel = build_kernel_array(parsed)
        if kernel is not None:
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
