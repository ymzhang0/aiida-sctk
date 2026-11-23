"""Auxiliary: Stoner I from the FSM runs of group sf/fsm/<MATERIAL> (launch_fsm.py).  usage: verdi run fsm_table.py MATERIAL ...
I = 1/N - dmu/m and 1/N - 4a, with N the DOS per spin at E_F from the m = 0 run itself (same cold smearing and k grid)."""
import sys, re, numpy as np
from aiida import orm
RY = 13.605693123
def w0(x): return np.exp(-np.minimum(200, (x - 1 / np.sqrt(2)) ** 2)) * (2 - np.sqrt(2) * x) / np.sqrt(np.pi)
for mat in sys.argv[1:]:
    runs = {}
    for n in orm.load_group(f'sf/fsm/{mat}').nodes:
        if not n.is_finished_ok: print(mat, n.pk, n.label, 'not finished ok', n.process_state, n.exit_status); continue
        tag = n.label.rsplit('_m', 1)[0]; m = float(n.label.rsplit('_m', 1)[1]); runs.setdefault(tag, {})[m] = n
    for tag, r in runs.items():
        if 0.0 not in r: continue
        n0 = r[0.0]; p0 = n0.outputs.output_parameters.get_dict(); sig = n0.inputs.pw.parameters.get_dict()['SYSTEM']['degauss']
        calc = n0.called[-1]; out = calc.outputs.retrieved.base.repository.get_object_content('aiida.out')
        b = n0.outputs.output_band; ev = b.get_bands(); w = b.get_array('weights'); ev = ev[0] if ev.ndim == 3 else ev
        ef = p0['fermi_energy'] if 'fermi_energy' in p0 else p0['fermi_energy_up']
        N = {s: float(np.sum(w[:, None] * w0((ef - ev) / (s * RY))) / (s * RY) / w.sum()) for s in (sig, 2 * sig)}
        print(f'== {tag}: N_cold(E_F) per spin = {N[sig]:.4f} /eV (sigma {sig} Ry), {N[2*sig]:.4f} (sigma {2*sig} Ry)')
        e0 = p0['energy']; ts0 = p0.get('energy_smearing', 0.0)
        for m in sorted(r):
            if m == 0: continue
            p = r[m].outputs.output_parameters.get_dict(); a = (p['energy'] - e0) / m ** 2
            o = r[m].called[-1].outputs.retrieved.base.repository.get_object_content('aiida.out')
            fu, fd = map(float, re.findall(r'spin up/dw Fermi energies are\s+([-0-9.]+)\s+([-0-9.]+)', o)[-1]); dmu = (fu - fd) / m
            print(f'   m={m}: dE={1e3*(p["energy"]-e0):.4f} meV, 4a={4*a:.4f} eV, dmu/m={dmu:.4f} eV, -TS change={1e3*(p.get("energy_smearing",0)-ts0):.4f} meV;'
                  f' I[1/N-4a]={1/N[sig]-4*a:.4f}, I[1/N-dmu/m]={1/N[sig]-dmu:.4f} eV')
