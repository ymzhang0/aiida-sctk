"""Tools for aiida-mcp (entry-point group ``aiida_mcp.tools``): plain functions, no import of aiida_mcp needed."""
from __future__ import annotations

from pathlib import Path


def average_sctk_kernels(pk: int) -> dict:
    """q-averaged SCTK kernels of a sctk.spin_fluctuation work chain: lambda_sf(i nu) and the screened Coulomb kernel mu_C(i nu) at nu = 0, 0.1 and 1 eV (star-weighted over the irreducible q points), the largest Stoner eigenvalue, and the fraction of the q grid covered. This is the convergence metric of the spin-fluctuation project."""
    from aiida_sctk.tools.kernels import _laddxc, average_kernels

    r = average_kernels(pk)
    return {
        'pk': r.node.pk, 'label': r.node.label, 'laddxc': _laddxc(r.node), 'number_of_qpoints': len(r.eig_max),
        'lambda_sf_at_0_0.1_1_eV': r.at(r.lambda_sf),
        'mu_coulomb_at_0_0.1_1_eV': None if r.mu_coulomb is None else r.at(r.mu_coulomb),
        'eig_max_q_average': float(r.weights @ r.eig_max), 'eig_max_largest': float(r.eig_max.max()),
        'bz_fraction': r.bz_fraction,
        'frequency_grid_eV': [float(r.frequencies[0]), float(r.frequencies[-1]), len(r.frequencies)],
    }


average_sctk_kernels.read_only = True


def write_epw_kernel_tables(pk: int, directory: str, name: str) -> dict:
    """Write the tables EPW reads for the isotropic Eliashberg equations from a sctk.spin_fluctuation work chain: `<directory>/<name>.lam_sf_iw` (fila2f_sf), `<name>.muc_iw` (fila2f_c, the Coulomb kernel of this run's laddxc) and `<name>.dos_fbw` (wide-window DOS for the full-bandwidth equations). Returns the paths, the static values and the DOS window with the number of electrons below E_F inside it (less than the valence count means the dos step missed the band bottom)."""
    from aiida_sctk.tools.kernels import write_epw_tables, write_fbw_dos

    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    sf, coulomb, dos = (str(base / f'{name}.{ext}') for ext in ('lam_sf_iw', 'muc_iw', 'dos_fbw'))
    r = write_epw_tables(pk, sf_path=sf, coulomb_path=coulomb)
    return {'lambda_sf_table': sf, 'mu_coulomb_table': coulomb, 'dos_file': dos, 'lambda_sf_static': float(r.lambda_sf[0]),
            'mu_coulomb_static': float(r.mu_coulomb[0]), 'dos': write_fbw_dos(pk, dos)}


write_epw_kernel_tables.read_only = True  # writes files, never the profile


def get_fsm_stoner_parameter(group: str, label_prefix: str = '') -> dict:
    """Stoner parameter I = 1/N(0) - 4a from the fixed-spin-moment scf runs of a group (PwBaseWorkChains labelled `<prefix>_m<moment>`, one of them with moment 0). N(0) is taken from the m = 0 run itself so that it is consistent with the smeared FSM energies."""
    from aiida import orm

    from aiida_sctk.tools.fsm import stoner_from_fsm

    runs = {}
    for node in orm.load_group(group).nodes:
        if node.is_finished_ok and '_m' in node.label and node.label.startswith(label_prefix):
            runs[float(node.label.rsplit('_m', 1)[1])] = node
    return stoner_from_fsm(runs)


get_fsm_stoner_parameter.read_only = True

tools = [average_sctk_kernels, write_epw_kernel_tables, get_fsm_stoner_parameter]
