#!/usr/bin/env python
"""Auxiliary: electronic DOS file in EPW's `prefix.dos` format from the tetrahedron DOS of an SCTK work chain,
for the isotropic full-bandwidth Eliashberg equations solved from fila2f (plugin input `fildos`).

usage: verdi run make_dos.py SPIN_FLUCTUATION_WORKCHAIN_PK OUT [EMIN EMAX [STEP]]   (window relative to E_F, eV; default -20 12 0.01)
Energies above the lowest energy of the highest nscf band are cut (the DOS is incomplete there); the cut is printed.
"""
import sys, re, numpy as np
from aiida import orm
pk, out = int(sys.argv[1]), sys.argv[2]
emin, emax = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (-20.0, 12.0)
step = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
w = orm.load_node(pk)
d = [c for c in w.called_descendants if c.process_label == 'DosCalculation'][-1]
txt = d.outputs.retrieved.base.repository.get_object_content('aiida.dos').splitlines()
ef = float(re.search(r'EFermi\s*=\s*([-0-9.]+)', txt[0]).group(1))
a = np.array([[float(x) for x in l.split()[:3]] for l in txt[1:]])
nscf = [c for c in w.called_descendants if c.process_label == 'PwCalculation' and c.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'nscf']
top = None
for c in nscf:
    if 'output_band' in c.outputs:
        b = c.outputs.output_band.get_bands(); b = b[0] if b.ndim == 3 else b; t = b[:, -1].min(); top = t if top is None else min(top, t)
lo = max(ef + emin, a[0, 0]); hi = min(ef + emax, a[-1, 0], top if top is not None else 1e9)
e = np.arange(lo, hi, step); dos = np.interp(e, a[:, 0], a[:, 1]); idos = np.cumsum(dos) * step
occ = e < ef; print(f'E_F = {ef} eV, window {lo - ef:+.2f} .. {hi - ef:+.2f} eV (highest band starts at {top - ef if top else float("nan"):+.2f}), '
                    f'N(E_F) = {np.interp(ef, a[:, 0], a[:, 1]):.4f} /eV (both spins), electrons below E_F in window = {dos[occ].sum() * step:.3f}, {len(e)} bins')
with open(out, 'w') as h:
    h.write(f'# EFermi[eV]   {ef:20.10E}   dos_EFermi[eV^-1]   {np.interp(ef, a[:, 0], a[:, 1]):20.10E}\n')
    h.write(f'# FermiWind[eV]   {max(ef - lo, hi - ef):20.10E}   Nr_electrons   {dos[occ].sum() * step:20.10E}\n')
    h.write(f'# dos_del[eV]   {step:20.10E}   Nr_bins   {len(e):10d}\n#            E [eV]           dos[state/eV]       Int dos[#]\n')
    for x, y, z in zip(e, dos, idos): h.write(f'{x:25.10E}{y:20.10E}{z:20.10E}\n')
