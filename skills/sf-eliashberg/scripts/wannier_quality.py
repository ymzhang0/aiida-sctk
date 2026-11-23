#!/usr/bin/env python
"""Wannier quality check (auxiliary): convergence, spreads and interpolated bands against DFT bands on the same path.

usage: verdi run wannier_quality.py W90_CALC_OR_WORKCHAIN_PK [...] [--dft 2861]
"""
import re, sys, numpy as np
from aiida import orm
args = [x for x in sys.argv[1:] if not x.startswith('--')]
dft_pk = int(sys.argv[sys.argv.index('--dft') + 1]) if '--dft' in sys.argv else 2861
if '--dft' in sys.argv: args.remove(str(dft_pk))
dft = orm.load_node(dft_pk).outputs.output_band; db = dft.get_bands()
for pk in args:
    node = orm.load_node(int(pk))
    calc = node if node.process_label == 'Wannier90Calculation' else [c for c in node.called_descendants if c.process_label == 'Wannier90Calculation'][-1]
    if not calc.is_finished or 'interpolated_bands' not in calc.outputs:
        print(pk, node.label, 'state', calc.process_state.value, calc.exit_status); continue
    p = calc.inputs.parameters.get_dict(); ef = p['fermi_energy']; nex = len(p.get('exclude_bands', []))
    out = calc.outputs.retrieved.base.repository.get_object_content('aiida.wout')
    dis = re.findall(r'^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-+\d.E]+)\s+[\d.]+\s+<-- DIS', out, re.M)
    conv = re.findall(r'<-- CONV', out); o = calc.outputs.output_parameters.get_dict()
    fs = out[out.rindex('Final State'):]; sp = [float(x) for x in re.findall(r'WF centre and spread\s+\d+\s+\(.*?\)\s+([\d.]+)', fs)]
    eb = calc.outputs.interpolated_bands.get_bands(); d = db[:, nex:nex + eb.shape[1]] if eb.shape[1] <= db.shape[1] - nex else None
    print(f"{pk} {node.label}: nwann {p['num_wann']}, bands {p['num_bands']}, froz EF{p.get('dis_froz_max', ef) - ef:+.2f}, "
          f"winmax {'none' if 'dis_win_max' not in p else f'EF{p['dis_win_max'] - ef:+.1f}'}; dis iters {len(dis)} (last delta {dis[-1][3] if dis else '-'}), "
          f"wan iters {len(conv)}; Omega_I {o.get('Omega_I'):.3f}, total {sum(sp):.3f}, max WF spread {max(sp):.3f} A^2; warnings {len(o.get('warnings', []))}")
    # each Wannier band is compared with the nearest DFT eigenvalue at the same k (band ordering can differ above the frozen window)
    allb = db[:, nex:]
    near = np.abs(eb[:, :, None] - allb[:, None, :]).min(axis=2)
    for lo, hi in ((-1, 1), (-0.4, 0.4), (1, p.get('dis_froz_max', ef) - ef)):
        m = (eb - ef > lo) & (eb - ef < hi)
        print(f'      EF{lo:+g}..{hi:+.2f} eV: max |dE| = {near[m].max() * 1e3:6.1f} meV, rms = {np.sqrt((near[m] ** 2).mean()) * 1e3:5.1f} meV ({m.sum()} points)')
