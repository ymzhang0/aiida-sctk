"""q average of the SCTK kernels and the tables EPW reads.

The kel calculation gives, for every irreducible q point, the Fermi-surface averaged kernels on an imaginary
frequency grid: ``mu_spin_fluctuation`` (lambda_sf) and ``mu_coulomb`` (mu_C), and the largest Stoner eigenvalue
``eig_max``.  The isotropic Eliashberg equations need their average over the Brillouin zone, i.e. over the
irreducible points weighted by the size of their stars.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

RY = 13.605693123  # eV


def star_weights(structure, mesh, shift, qcart) -> np.ndarray:
    """Number of grid points in the star of each SCTK q point.

    ``qcart`` are the q points as SCTK prints them (Cartesian, units 2 pi / alat); ``mesh`` and ``shift`` (0 or 1 per
    direction, 1 = half-grid shift) define the uniform grid.  The stars are the orbits spglib finds for that grid
    with time reversal.  Raises if two q points share an orbit or if the stars do not fill the grid.
    """
    import spglib

    atoms = structure.get_ase()
    cell = atoms.get_cell().array
    alat = np.linalg.norm(cell[0])
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        mesh, (cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
        is_shift=shift, is_time_reversal=True, symprec=1e-5)
    mesh = np.array(mesh)
    shift = np.array(shift)
    lookup = {tuple((2 * g + shift) % (2 * mesh)): i for i, g in enumerate(grid)}
    counts = np.bincount(mapping, minlength=len(mapping))
    weights, seen = [], set()
    for q in qcart:
        frac = np.array(q) @ cell.T / alat  # crystal coordinates
        key = tuple(np.rint(frac * 2 * mesh).astype(int) % (2 * mesh))
        orbit = mapping[lookup[key]]
        if orbit in seen:
            raise ValueError('two SCTK q points fall in the same orbit')
        seen.add(orbit)
        weights.append(counts[orbit])
    if sum(weights) != mesh.prod():
        raise ValueError(f'the stars of the q points cover {sum(weights)} of {mesh.prod()} grid points')
    return np.array(weights, float)


@dataclass
class KernelAverage:
    """q-averaged kernels of one spin-fluctuation work chain."""

    node: object            #: the work chain
    frequencies: np.ndarray  #: imaginary frequencies nu [eV]
    lambda_sf: np.ndarray   #: q-averaged lambda_sf(i nu)
    mu_coulomb: np.ndarray | None  #: q-averaged mu_C(i nu); None when the kernels had to be merged without it
    eig_max: np.ndarray     #: largest Stoner eigenvalue of each computed q point
    weights: np.ndarray     #: normalised star weights of the computed q points
    bz_fraction: float      #: fraction of the grid the computed q points cover (below 1 when Gamma is skipped)

    def at(self, values, energies=(0.0, 0.1, 1.0)) -> list[float]:
        """``values`` (on the frequency grid) linearly interpolated at ``energies`` [eV]."""
        return [float(x) for x in np.interp(energies, self.frequencies, values)]


def average_kernels(node) -> KernelAverage:
    """Star-weighted q average of the kernels of a ``sctk.spin_fluctuation`` work chain.

    Uses the merged ``kernel`` output; for a work chain that did not finish cleanly the kernels of its kel
    calculations are concatenated instead, provided together they cover every q point.
    """
    from aiida import orm

    node = orm.load_node(node) if not isinstance(node, orm.Node) else node
    names = ('qpoint_indices', 'eig_max', 'mu_spin_fluctuation', 'mu_coulomb')
    if 'kernel' in node.outputs:
        kernel = node.outputs.kernel
        par = node.outputs.output_parameters.get_dict()
        arrays = {n: kernel.get_array(n) for n in names if n in kernel.get_arraynames()}
        frequencies = kernel.get_array('frequencies')
    else:
        calcs = [c for c in node.called_descendants if c.process_label == 'SCTKCalculation' and 'kernel' in c.outputs]
        if not calcs:
            raise ValueError(f'process {node.pk} has no kernel output')
        par = calcs[0].outputs.output_parameters.get_dict()
        common = [n for n in names if all(n in c.outputs.kernel.get_arraynames() for c in calcs)]
        cat = {n: np.concatenate([c.outputs.kernel.get_array(n) for c in calcs]) for n in common}
        order = np.argsort(cat['qpoint_indices'])
        if not len(set(cat['qpoint_indices'])) == len(order) == par['number_of_qpoints']:
            raise ValueError('incomplete q coverage')
        arrays = {n: v[order] for n, v in cat.items()}
        frequencies = calcs[0].outputs.kernel.get_array('frequencies')
    w_all = star_weights(node.inputs.structure, par['q_grid'], par['q_shift'], par['qpoints_cartesian'])
    idx = arrays['qpoint_indices'] - 1
    w = w_all[idx] / w_all[idx].sum()
    return KernelAverage(
        node=node, frequencies=frequencies * RY, lambda_sf=w @ arrays['mu_spin_fluctuation'],
        mu_coulomb=w @ arrays['mu_coulomb'] if 'mu_coulomb' in arrays else None,
        eig_max=arrays['eig_max'], weights=w, bz_fraction=float(w_all[idx].sum() / w_all.sum()))


def write_epw_qtable(node, path) -> dict:
    """Write the per-q kernels of a ``sctk.spin_fluctuation`` work chain as EPW's per-q table (``fila2f_sf``).

    EPW does the star-weighted q average itself (header word ``qtable``).  Format, free format::

        # qtable imag ...                         header
        nq  nfreq
        nu(1:nfreq)                               [meV], nu(1) = 0
        per q point:  xq(1:3) (crystal)  w_q  lambda_sf(inf)  mu_C(inf)
                      lambda_sf(q, nu(1:nfreq))
                      mu_C(q, nu(1:nfreq))

    ``w_q`` is the multiplicity of the star of the q point (spglib, as in :func:`star_weights`); EPW cannot
    compute it in this mode because it does not load the structure.  Returns a short summary.
    """
    from aiida import orm

    node = orm.load_node(node) if not isinstance(node, orm.Node) else node
    if 'kernel' not in node.outputs:
        raise ValueError(f'process {node.pk} has no merged kernel output')
    kernel = node.outputs.kernel
    par = node.outputs.output_parameters.get_dict()
    names = kernel.get_arraynames()
    need = ('qpoint_indices', 'mu_spin_fluctuation', 'mu_coulomb', 'mu_coulomb_infinite_frequency')
    missing = [n for n in need if n not in names]
    if missing:
        raise ValueError(f'kernel output of {node.pk} lacks {missing}')
    idx = kernel.get_array('qpoint_indices') - 1
    w_all = star_weights(node.inputs.structure, par['q_grid'], par['q_shift'], par['qpoints_cartesian'])
    qcart = np.array(par['qpoints_cartesian'], float)
    atoms = node.inputs.structure.get_ase(); cell = atoms.get_cell().array; alat = np.linalg.norm(cell[0])
    frac = qcart @ cell.T / alat
    nu = kernel.get_array('frequencies') * RY * 1000.0       # Ry -> meV
    sf = kernel.get_array('mu_spin_fluctuation'); mc = kernel.get_array('mu_coulomb')
    cinf = kernel.get_array('mu_coulomb_infinite_frequency')
    sfinf = kernel.get_array('mu_spin_fluctuation_infinite_frequency') if 'mu_spin_fluctuation_infinite_frequency' in names \
        else np.zeros(len(idx))
    with open(path, 'w') as handle:
        handle.write(f'# qtable imag  SCTK per-q Fermi-surface averaged kernels, work chain {node.pk} {node.uuid}, '
                     f'laddxc = {_laddxc(node)}\n')
        handle.write(f'{len(idx)} {len(nu)}\n')
        handle.write(' '.join(f'{x:.6f}' for x in nu) + '\n')
        for i, iq in enumerate(idx):
            handle.write(' '.join(f'{x:.10f}' for x in frac[iq]) + f' {w_all[iq]:.1f} {sfinf[i]:.8e} {cinf[i]:.8e}\n')
            handle.write(' '.join(f'{x:.8e}' for x in sf[i]) + '\n')
            handle.write(' '.join(f'{x:.8e}' for x in mc[i]) + '\n')
    w = w_all[idx]
    return {'nq': int(len(idx)), 'nfreq': int(len(nu)), 'max_freq_ev': float(nu[-1] / 1000),
            'lambda_sf_0': float(w @ sf[:, 0] / w.sum()), 'mu_c_0': float(w @ mc[:, 0] / w.sum()),
            'mu_c_inf': float(w @ cinf / w.sum()), 'weights': w.tolist()}


def _laddxc(node) -> int | None:
    try:
        return node.inputs.sctk.sctk.parameters.get_dict()['KEL']['laddxc']
    except (AttributeError, KeyError):
        return None


def write_epw_tables(node, sf_path=None, coulomb_path=None) -> KernelAverage:
    """Write the q-averaged kernels as the tables EPW reads (``fila2f_sf`` and ``fila2f_c``).

    Format: one header line containing the word ``imag``, then ``nu[meV]  value`` starting at nu = 0.
    """
    result = average_kernels(node)
    source = f'SCTK, star-weighted q average, work chain {result.node.pk} {result.node.uuid}'
    if sf_path:
        with open(sf_path, 'w') as handle:
            handle.write(f'# imag-axis table: nu[meV]  lambda_sf(i nu)   ({source})\n')
            for x, y in zip(result.frequencies * 1000, result.lambda_sf):
                handle.write(f'{x:16.6f} {y:16.8e}\n')
    if coulomb_path:
        if result.mu_coulomb is None:
            raise ValueError('no mu_coulomb array to write')
        with open(coulomb_path, 'w') as handle:
            handle.write(f'# imag-axis table: nu[meV]  mu_C(i nu)   (SCTK, star-weighted q average, laddxc = {_laddxc(result.node)}, '
                         f'work chain {result.node.pk} {result.node.uuid})\n')
            for x, y in zip(result.frequencies * 1000, result.mu_coulomb):
                handle.write(f'{x:16.6f} {y:16.8e}\n')
    return result


def write_fbw_dos(node, path, emin=-20.0, emax=12.0, step=0.01) -> dict:
    """Write the tetrahedron DOS of the work chain's dos step in EPW's ``prefix.dos`` format.

    EPW's own DOS file only covers the Fermi window of the a2F run; the isotropic full-bandwidth equations with a
    Coulomb kernel need the whole band.  The window [emin, emax] is relative to E_F [eV] and is cut at the range
    of the dos step and below the lowest energy of the highest nscf band (above it the DOS is incomplete).
    Returns the window actually written, N(E_F) and the number of electrons below E_F inside the window - if that
    is less than the number of valence electrons the dos step started above the band bottom.
    """
    from aiida import orm

    node = orm.load_node(node) if not isinstance(node, orm.Node) else node
    dos = [c for c in node.called_descendants if c.process_label == 'DosCalculation'][-1]
    lines = dos.outputs.retrieved.base.repository.get_object_content('aiida.dos').splitlines()
    fermi = float(re.search(r'EFermi\s*=\s*([-0-9.]+)', lines[0]).group(1))
    table = np.array([[float(x) for x in line.split()[:3]] for line in lines[1:]])
    top = None
    for calc in node.called_descendants:
        if calc.process_label == 'PwCalculation' and calc.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'nscf' \
                and 'output_band' in calc.outputs:
            bands = calc.outputs.output_band.get_bands()
            bands = bands[0] if bands.ndim == 3 else bands
            lowest = bands[:, -1].min()
            top = lowest if top is None else min(top, lowest)
    low = max(fermi + emin, table[0, 0])
    high = min(fermi + emax, table[-1, 0], top if top is not None else np.inf)
    energies = np.arange(low, high, step)
    values = np.interp(energies, table[:, 0], table[:, 1])
    integral = np.cumsum(values) * step
    dos_fermi = float(np.interp(fermi, table[:, 0], table[:, 1]))
    electrons = float(values[energies < fermi].sum() * step)
    with open(path, 'w') as handle:
        handle.write(f'# EFermi[eV]   {fermi:20.10E}   dos_EFermi[eV^-1]   {dos_fermi:20.10E}\n')
        handle.write(f'# FermiWind[eV]   {max(fermi - low, high - fermi):20.10E}   Nr_electrons   {electrons:20.10E}\n')
        handle.write(f'# dos_del[eV]   {step:20.10E}   Nr_bins   {len(energies):10d}\n#            E [eV]           dos[state/eV]       Int dos[#]\n')
        for x, y, z in zip(energies, values, integral):
            handle.write(f'{x:25.10E}{y:20.10E}{z:20.10E}\n')
    return {'fermi_energy': fermi, 'window': [float(low - fermi), float(high - fermi)], 'dos_at_fermi_both_spins': dos_fermi,
            'electrons_below_fermi_in_window': electrons, 'bins': len(energies),
            'highest_band_starts_at': None if top is None else float(top - fermi)}
