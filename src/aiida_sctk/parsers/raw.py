"""Plain-Python readers for the text output of ``sctk.x`` (calculation='kel').

No AiiDA import: every function takes a path, an open text file or the text itself and returns
plain dictionaries / lists, so that it can be used on files outside the provenance as well.
"""
import os
import re


def read_text(source):
    """Return the text of ``source``: a path, an open text file, or the text itself."""
    if hasattr(source, 'read'):
        text = source.read()
        return text.decode() if isinstance(text, bytes) else text
    if isinstance(source, os.PathLike) or (isinstance(source, str) and '\n' not in source and os.path.isfile(source)):
        with open(source, encoding='utf-8') as handle:
            return handle.read()
    if isinstance(source, bytes):
        return source.decode()
    return source


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


def parse_kel_stdout(source):
    """Parse the summary that ``sctk.x`` (calculation='kel') prints to stdout.

    :param source: path, open text file, or the text of the standard output.
    :return: dictionary with the q grid, the list of irreducible q points (Cartesian, 2 pi / alat),
        the Chebyshev imaginary frequencies [Ry] and, for every computed q point, the Stoner
        diagnostics and the Fermi-surface averaged Coulomb / spin-fluctuation kernels.
    """
    text = read_text(source)
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


def parse_head_file(source):
    """Parse a ``<prefix>.pi0_head.dat`` / ``ixc_pi0_head.dat`` / ``chis_head.dat`` file.

    These files hold the G = G' = 0 element of a response matrix, one block per q point and one
    line per frequency.  The columns after the frequency are (Re, Im) pairs: one pair for the
    dimensionless ``ixc_pi0_head``; two pairs, [1/(Ry bohr^3)] and [1/eV per cell], for
    ``pi0_head`` and ``chis_head`` (both spins summed, negative sign).  ``chis_head`` has a leading
    spin-channel column.

    :param source: path, open text file, or the text of the file.
    :return: ``{'header': [...], 'real_frequency': bool, 'qpoints': [{'index', 'q_crystal',
        'q_cartesian', 'channel' (chis only), 'frequency_index', 'frequency_ry', 'frequency_ev',
        'values': [[complex, ...] per frequency]}]}``; ``values[i][-1]`` is the entry in the
        last unit of the file (1/eV per cell, or dimensionless).
    """
    header, blocks, current = [], [], None
    for line in read_text(source).splitlines():
        stripped = line.strip()
        if stripped.startswith('# iq'):
            numbers = re.findall(_FLOAT, stripped.split('q_crys')[1])
            current = {'index': int(stripped.split()[2]), 'q_crystal': [_to_float(x) for x in numbers[:3]],
                       'q_cartesian': [_to_float(x) for x in numbers[-3:]], 'frequency_index': [],
                       'frequency_ry': [], 'frequency_ev': [], 'values': []}
            blocks.append(current)
        elif stripped.startswith('#'):
            header.append(stripped.lstrip('# '))
        elif stripped and current is not None:
            tokens = stripped.split()
            lead = 1 if len(tokens) % 2 == 1 else 2               # even count: chis_head has a leading spin-channel column
            if lead == 2:
                channel = int(tokens[0])
                if current.setdefault('channel', channel) != channel:
                    current = dict(current, channel=channel, frequency_index=[], frequency_ry=[], frequency_ev=[], values=[])
                    blocks.append(current)
            current['frequency_index'].append(int(tokens[lead - 1]))
            current['frequency_ry'].append(_to_float(tokens[lead]))
            current['frequency_ev'].append(_to_float(tokens[lead + 1]))
            numbers = [_to_float(x) for x in tokens[lead + 2:]]
            current['values'].append([complex(re_, im_) for re_, im_ in zip(numbers[0::2], numbers[1::2])])
    return {'header': header, 'real_frequency': any('REAL-FREQUENCY' in h for h in header), 'qpoints': blocks}

