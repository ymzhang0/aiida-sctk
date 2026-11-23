"""Stoner parameter from fixed-spin-moment (FSM) total energies."""
from __future__ import annotations

import re

import numpy as np

RY = 13.605693123  # eV


def _cold_delta(x):
    """Marzari-Vanderbilt cold-smearing delta function (QE ``w0gauss`` with ngauss = -1)."""
    return np.exp(-np.minimum(200, (x - 1 / np.sqrt(2)) ** 2)) * (2 - np.sqrt(2) * x) / np.sqrt(np.pi)


def stoner_from_fsm(runs: dict) -> dict:
    """Stoner I = 1/N(0) - 4a from scf runs at fixed total magnetisation.

    ``runs`` maps the moment m [mu_B per cell] to a finished ``PwBaseWorkChain`` with ``tot_magnetization = m``; m = 0
    must be present.  E(m) - E(0) = a m^2 (+ b m^4).  N(0) is the DOS per spin at E_F **of the m = 0 run itself**,
    with its own cold smearing and k grid: the FSM curvature measures the smeared susceptibility, so a tetrahedron
    N(0) from another calculation is inconsistent with it (errors of 0.2-0.8 eV were seen).
    Returns N(0), and I from each moment (energy curvature and Fermi-level splitting) and extrapolated to m -> 0.
    """
    zero = runs[0.0]
    parameters = zero.outputs.output_parameters.get_dict()
    sigma = zero.inputs.pw.parameters.get_dict()['SYSTEM']['degauss']
    band = zero.outputs.output_band
    energies = band.get_bands()
    energies = energies[0] if energies.ndim == 3 else energies
    weights = band.get_array('weights')
    fermi = parameters.get('fermi_energy', parameters.get('fermi_energy_up'))
    n0 = float(np.sum(weights[:, None] * _cold_delta((fermi - energies) / (sigma * RY))) / (sigma * RY) / weights.sum())
    result = {'dos_per_spin': n0, 'degauss_ry': sigma, 'moments': {}}
    e0 = parameters['energy']
    curvature = {}
    for m in sorted(x for x in runs if x != 0.0):
        p = runs[m].outputs.output_parameters.get_dict()
        out = runs[m].called[-1].outputs.retrieved.base.repository.get_object_content('aiida.out')
        up, down = map(float, re.findall(r'spin up/dw Fermi energies are\s+([-0-9.]+)\s+([-0-9.]+)', out)[-1])
        a = (p['energy'] - e0) / m ** 2
        curvature[m] = a
        result['moments'][m] = {'delta_e_mev': 1e3 * (p['energy'] - e0), 'four_a_ev': 4 * a, 'i_from_energy_ev': 1 / n0 - 4 * a,
                                'i_from_fermi_splitting_ev': 1 / n0 - (up - down) / m}
    if len(curvature) >= 2:  # dE = a m^2 + b m^4 through the two smallest moments: a is the m -> 0 curvature
        m1, m2 = sorted(curvature)[:2]
        a0 = (curvature[m1] * m2 ** 2 - curvature[m2] * m1 ** 2) / (m2 ** 2 - m1 ** 2)
        result['i_extrapolated_ev'] = 1 / n0 - 4 * a0
    return result
