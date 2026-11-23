#!/usr/bin/env python
"""Table of the fine-grid a2F tests (auxiliary): DOS at E_F, lambda, omega_log, Allen-Dynes Tc, wall time.
usage: verdi run epw_grid_table.py PK [PK ...]   (EpwBaseWorkChain pks of the a2f stage)"""
import re, sys, numpy as np
from aiida import orm
print(f"{'pk':>6} {'label':34} {'state':>9} {'DOS/spin':>9} {'lambda':>8} {'wlog meV':>9} {'Tc_AD K':>8} {'wall':>10}")
for pk in map(int, sys.argv[1:]):
    wc = orm.load_node(pk); calcs = [x for x in wc.called if x.process_label == 'EpwCalculation']
    state = wc.process_state.value + (f'[{wc.exit_status}]' if wc.exit_status is not None else '')
    if not calcs or 'retrieved' not in calcs[-1].outputs:
        print(f'{pk:>6} {wc.label[:34]:34} {state:>9}'); continue
    r = calcs[-1].outputs.retrieved.base.repository; out = r.get_object_content('aiida.out')
    g = lambda pat: (re.findall(pat, out) or [None])[-1]
    dos = g(r'DOS\(states/spin/eV/Unit Cell\) =\s+([0-9.E+-]+)'); lam = g(r'Electron-phonon coupling strength =\s+([0-9.]+)')
    tc = g(r'Allen-Dynes modified McMillan expression =\s+([0-9.]+)'); wall = g(r'EPW\s+:\s+\S+\s*\S* CPU\s+(\S+\s?\S*) WALL')
    wlog = None
    if 'aiida.a2f' in r.list_object_names():
        rows = []
        for line in r.get_object_content('aiida.a2f').splitlines():
            t = line.split()
            try: rows.append((float(t[0]), float(t[1])))
            except (ValueError, IndexError): pass
        a = np.array(rows); w, f = a[:, 0], a[:, 1]; m = (w > 0) & (f > 0)
        l0 = 2 * np.trapezoid(f[m] / w[m], w[m]); wlog = np.exp(2 / l0 * np.trapezoid(f[m] / w[m] * np.log(w[m]), w[m]))
    fmt = lambda x, n: f'{float(x):>{n}.4f}' if x is not None else ' ' * (n - 1) + '-'
    print(f'{pk:>6} {wc.label[:34]:34} {state:>9} {fmt(dos, 9)} {fmt(lam, 8)} {fmt(wlog, 9)} {fmt(tc, 8)} {wall or "-":>10}')
