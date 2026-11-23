#!/usr/bin/env python
"""Result figures of one material for the record page (auxiliary; not part of the provenance).

usage: plot_results.py MATERIAL OUTDIR
  <M>_prod_kernels         (a) lambda_sf(i nu), (b) screened Coulomb kernel mu_C(i nu) for laddxc = 0 / 1 / 2 (production runs)
  <M>_a2f_conv             (a) a2F(w) on the fine grids tested, (b) lambda_ph against the fine grid
  <M>_tc_linear            largest eigenvalue of the linearised isotropic equation against T, all levels of approximation
  <M>_gap_const_dos        Z(i w) and Delta(i w), constant-DOS non-linear solver, one temperature
  <M>_gap_fbw              Z(i w) and Delta(i w), full bandwidth, one temperature
  <M>_gap_fbw_T            Delta(i w_0) against T for the full-bandwidth runs
Entries of CFG that are missing are skipped.
"""
import os, re, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from aiida import orm, load_profile
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figure-style'))
import paperfig as pf
import qavg
load_profile('sf')
L = ['el-ph, $\\mu^*$ = 0.13', '+ SF, $\\mu^*$ = 0.13', '+ SF + $\\mu_C$ (RPA)', '+ SF + $\\mu_C$ (KO)']      # the four levels of approximation
CFG = {
    'V': dict(prod={0: 8821, 1: 18760, 2: 16496},
              a2f=[('k 24, q 24', 14648), ('k 36, q 12', 14883), ('k 48, q 12', 14896), ('k 60, q 12', 14909), ('k 48, q 24', 15045), ('k 60, q 20', 15343), ('k 60, q 20, 16 WF', 16542)],
              linear=[('el-ph, $\\mu^*$ = 0.13', 18287), ('+ SF, $\\mu^*$ = 0.13', 18300), ('+ SF + $\\mu_C$ (RPA)', 16998), ('+ SF + $\\mu_C$ (KO)', 18324)],
              const=[('el-ph, $\\mu^*$ = 0.13', 20880), ('+ SF, $\\mu^*$ = 0.13', 20893), ('+ SF + $\\mu_C$ (RPA)', 20908), ('+ SF + $\\mu_C$ (KO)', 20922)],
              fbw=[('el-ph, $\\mu^*$ = 0.13', 20936), ('+ SF, $\\mu^*$ = 0.13', 20949), ('+ SF + $\\mu_C$ (KO)', 20963)],
              fbw_T=[('el-ph, $\\mu^*$ = 0.13', 19440), ('+ SF, $\\mu^*$ = 0.13', 19455), ('+ SF + $\\mu_C$ (KO), cutoff 60 eV', 19747),
                     ('cutoff 120 eV', 19763), ('cutoff 240 eV', 19777)], tc_exp=5.4),
    'Nb': dict(prod={0: 8852, 1: 18791, 2: 18353},
               a2f=[('k 24, q 12', 15056), ('k 36, q 12', 15069), ('k 48, q 12', 15083), ('k 60, q 12', 15098), ('k 48, q 24', 15125), ('k 60, q 20', 15356)],
               linear=[(L[0], 17024), (L[1], 17035), (L[2], 20987), (L[3], 20225)], const=[(L[0], 20999), (L[1], 21012), (L[2], 21024), (L[3], 21042)],
               fbw=[(L[0], 21056), (L[1], 21071), (L[3], 21085)],
               fbw_T=[(L[0], 20239), (L[1], 20254), (L[3] + ', cutoff 60 eV', 20270), ('cutoff 120 eV', 20287), ('cutoff 240 eV', 20303)], tc_exp=9.2),
    'MgB2': dict(prod={0: 13662, 1: 18813, 2: 18882},
                 a2f=[('k 36x36x24, q 12x12x8', 18521), ('k 48x48x32, q 12x12x8', 18532), ('k 60x60x40, q 12x12x8', 18556), ('k 48x48x32, q 24x24x16', 18688), ('k 72x72x48, q 24x24x16', 18711)],
                 linear=[(L[0], 19792), (L[1], 19805), (L[2], 21097), (L[3], 20624)], const=[(L[0], 21107), (L[1], 21124), (L[2], 21143), (L[3], 21158)],
                 fbw=[(L[0], 21172), (L[1], 21185), (L[3], 21199)],
                 fbw_T=[(L[0], 20645), (L[1], 20660), (L[3] + ', cutoff 60 eV', 20674), ('cutoff 120 eV', 20688), ('cutoff 240 eV', 20702)], tc_exp=39.0),
    'Pb': dict(prod={0: 8874, 1: 18843, 2: 18916},
               a2f=[('k 36, q 12', 20736), ('k 48, q 12', 20749), ('k 60, q 12', 20762), ('k 48, q 24', 20776), ('k 60, q 20', 20789)], tc_exp=7.2),
}
material, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); cfg = CFG[material]
STY = [('black', 'o', '-'), ('blue', 's', '--'), ('green', '^', '-.'), ('orange', 'D', ':'), ('purple', 'v', '-'), ('sky', 'P', '--'), ('yellow', 'X', '-.')]
def pair():
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={'wspace': 0.27})
    for axis in axes: pf.style_axis(axis)
    return figure, axes
def save(figure, tag):
    base = os.path.join(out, f'{material}_{tag}'); figure.savefig(base + '.svg', bbox_inches='tight', pad_inches=0.02); print('wrote', pf.save(figure, base + '.pdf'))
def epw_calc(pk):
    wc = orm.load_node(pk); cs = [x for x in wc.called if x.process_label == 'EpwCalculation']
    return cs[-1] if cs and 'retrieved' in cs[-1].outputs else None
def content(calc, name): return calc.outputs.retrieved.base.repository.get_object_content(name)
def imag_files(calc):
    """{T: array(w, Z, Delta[, shift])} from the aiida.imag_iso_* files."""
    res = {}
    for n in calc.outputs.retrieved.base.repository.list_object_names():
        m = re.match(r'aiida\.imag_iso_([0-9.]+)$', n)
        if m: res[float(m.group(1))] = np.array([[float(x) for x in l.split()] for l in content(calc, n).splitlines()[1:] if l.strip()])
    return res

with pf.style('single'), plt.rc_context({'svg.fonttype': 'path'}):
    if 'prod' in cfg:
        figure, axes = pair()
        node, nu, lam, eig, w, frac = qavg.analyse(cfg['prod'][max(cfg['prod'])]); x = np.where(nu > 0, nu, np.nan)
        sf = node.outputs.kernel.get_array('mu_spin_fluctuation')
        for row in sf: axes[0].plot(x, row, color='0.75', linewidth=0.7)
        axes[0].plot(x, lam, color=pf.color('blue'), marker='o', markersize=4, label=r'$q$ average')
        axes[0].plot([], [], color='0.75', linewidth=0.7, label=f'{len(sf)} $q$ points'); pf.reference_line(axes[0], lam[0])
        axes[0].plot([], [], color='black', linestyle='--', linewidth=0.9, label=rf'static value {lam[0]:.3f}')
        axes[0].set_xscale('log'); axes[0].set_xlabel(r'$\nu$ (eV)'); axes[0].set_ylabel(r'$\lambda^{\mathrm{sf}}(i\nu)$'); axes[0].set_ylim(bottom=0)
        axes[0].legend(loc='upper right', fontsize=10); pf.panel_label(axes[0], material, index=0, loc='lower left')
        for (lx, pk), (c, mk, ls), lab in zip(sorted(cfg['prod'].items()), [STY[0], STY[1], STY[3]], ['RPA (laddxc 0)', 'laddxc 1', 'KO (laddxc 2)']):
            node, nu, lam, eig, w, frac = qavg.analyse(pk); mu = w @ node.outputs.kernel.get_array('mu_coulomb')
            axes[1].plot(np.where(nu > 0, nu, np.nan), mu, color=pf.color(c), marker=mk, markersize=4, linestyle=ls, label=f'{lab}: static {mu[0]:.3f}')
            print(f'prod laddxc {lx} pk {pk}: mu_C(0) = {mu[0]:.4f}, lambda_sf(0) = {lam[0]:.4f}')
        axes[1].set_xscale('log'); axes[1].set_xlabel(r'$\nu$ (eV)'); axes[1].set_ylabel(r'$\mu^{C}(i\nu)$'); axes[1].set_ylim(0, None)
        axes[1].legend(loc='upper left', fontsize=10); pf.panel_label(axes[1], material, index=1, loc='lower right'); save(figure, 'prod_kernels')

    if 'a2f' in cfg:
        figure, axes = pair(); labs, lams = [], []
        for (lab, pk), (c, mk, ls) in zip(cfg['a2f'], STY):
            calc = epw_calc(pk)
            if calc is None: continue
            rows = []
            for l in content(calc, 'aiida.a2f').splitlines():
                t = l.split()
                try: rows.append((float(t[0]), float(t[1])))
                except (ValueError, IndexError): pass
            a = np.array(rows); lam = float(re.findall(r'Electron-phonon coupling strength =\s+([0-9.]+)', content(calc, 'aiida.out'))[-1])
            # the a2F of the file has the smallest phonon smearing (0.05 meV): smooth with a 0.5 meV Gaussian for the plot only
            dw = a[1, 0] - a[0, 0]; k = np.exp(-0.5 * (np.arange(-40, 41) * dw / 0.5) ** 2); k /= k.sum()
            axes[0].plot(a[:, 0], np.convolve(a[:, 1], k, mode='same'), color=pf.color(c), linestyle=ls, label=lab); labs.append(lab); lams.append(lam)
        axes[0].set_xlabel(r'$\omega$ (meV)'); axes[0].set_ylabel(r'$\alpha^2F(\omega)$'); axes[0].set_ylim(bottom=0); axes[0].set_xlim(left=0)
        axes[0].legend(loc='upper left', fontsize=8.5); pf.panel_label(axes[0], material, index=0, loc='upper right')
        for i, ((c, mk, ls), lam) in enumerate(zip(STY, lams)): axes[1].plot([i], [lam], color=pf.color(c), marker=mk, linestyle='none', markersize=8)
        axes[1].plot(range(len(lams)), lams, color='0.6', linewidth=0.8, zorder=0)
        axes[1].set_xticks(range(len(labs))); axes[1].set_xticklabels([l.replace(', ', '\n', 1) for l in labs], fontsize=8.5); axes[1].set_ylabel(r'$\lambda^{\mathrm{ph}}$')
        pf.panel_label(axes[1], material, index=1, loc='upper right'); save(figure, 'a2f_conv')

    if 'linear' in cfg:
        figure, axis = plt.subplots(figsize=(5.6, 3.9)); pf.style_axis(axis)
        for (lab, pk), (c, mk, ls) in zip(cfg['linear'], STY):
            calc = epw_calc(pk)
            if calc is None: continue
            o = content(calc, 'aiida.out'); o = o[o.index('Max.'):] if 'Max.' in o else ''
            d = np.array([[float(a), float(b)] for a, b in re.findall(r'^\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s+[0-9.]+\s+\d+\s*$', o, re.M)])
            if not len(d): continue
            i = np.where((d[:-1, 1] >= 1) & (d[1:, 1] < 1))[0]; tc = d[i[0], 0] + (d[i[0], 1] - 1) / (d[i[0], 1] - d[i[0] + 1, 1]) * (d[i[0] + 1, 0] - d[i[0], 0]) if len(i) else np.nan
            axis.plot(d[:, 0], d[:, 1], color=pf.color(c), linestyle=ls, label=rf'{lab}: $T_c$ = {tc:.1f} K'); axis.plot([tc], [1], color=pf.color(c), marker=mk, linestyle='none')
            print(f'linear {lab} pk {pk}: Tc = {tc:.2f} K')
        pf.reference_line(axis, 1.0)
        if 'tc_exp' in cfg: axis.axvline(cfg['tc_exp'], color='0.5', linewidth=0.8, linestyle=':'); axis.text(cfg['tc_exp'], 0.03, ' exp.', transform=axis.get_xaxis_transform(), fontsize=9, color='0.4')
        axis.set_xlabel(r'$T$ (K)'); axis.set_ylabel('Largest eigenvalue'); axis.set_ylim(0, 3.0); axis.set_xlim(0, 40); axis.legend(loc='upper right', fontsize=9.5)
        pf.panel_label(axis, material, index=0, loc='lower left'); save(figure, 'tc_linear')

    for key, tag in (('const', 'gap_const_dos'), ('fbw', 'gap_fbw')):
        if key not in cfg: continue
        figure, axes = pair(); T0 = None
        for (lab, pk), (c, mk, ls) in zip(cfg[key], STY):
            calc = epw_calc(pk)
            if calc is None: print(key, lab, pk, 'not finished'); continue
            f = imag_files(calc)
            if not f: print(key, lab, pk, 'no imag_iso file'); continue
            T0 = min(f); d = f[T0]; m = d[:, 0] < 0.5
            axes[0].plot(d[m, 0] * 1e3, d[m, 1], color=pf.color(c), linestyle=ls, label=lab); axes[1].plot(d[m, 0] * 1e3, d[m, 2] * 1e3, color=pf.color(c), linestyle=ls, label=lab)
            print(f'{key} {lab} pk {pk}: T = {T0} K, Z(w0) = {d[0, 1]:.3f}, Delta(w0) = {d[0, 2] * 1e3:.3f} meV')
        for axis, yl in zip(axes, (r'$Z(i\omega)$', r'$\Delta(i\omega)$ (meV)')):
            axis.set_xscale('log'); axis.set_xlabel(r'$\omega$ (meV)'); axis.set_ylabel(yl)
        axes[1].axhline(0, color='0.5', linewidth=0.7); axes[0].legend(loc='upper right', fontsize=9.5)
        pf.panel_label(axes[0], f'{material}, {T0:g} K' if T0 else material, index=0, loc='lower left'); pf.panel_label(axes[1], f'{material}, {T0:g} K' if T0 else material, index=1, loc='upper right'); save(figure, tag)

    if 'fbw_T' in cfg:
        figure, axis = plt.subplots(figsize=(5.6, 3.9)); pf.style_axis(axis)
        for (lab, pk), (c, mk, ls) in zip(cfg['fbw_T'], STY):
            calc = epw_calc(pk)
            if calc is None: continue
            f = imag_files(calc); T = np.array(sorted(f)); D = np.array([f[t][0, 2] * 1e3 for t in T])
            tc = T[-1] + (T[-1] - T[-2]) * D[-1] ** 2 / (D[-2] ** 2 - D[-1] ** 2) if len(T) > 1 and D[-2] > D[-1] else np.nan      # Delta^2 linear in T near Tc
            axis.plot(T, D, color=pf.color(c), marker=mk, markersize=4, linestyle=ls, label=rf'{lab}: $T_c$ = {tc:.1f} K'); print(f'fbw_T {lab} pk {pk}: Tc = {tc:.2f} K')
        if 'tc_exp' in cfg: axis.axvline(cfg['tc_exp'], color='0.5', linewidth=0.8, linestyle=':')
        axis.set_xlabel(r'$T$ (K)'); axis.set_ylabel(r'$\Delta(i\omega_0)$ (meV)'); axis.set_ylim(0, axis.get_ylim()[1] * 1.55); axis.set_xlim(left=0); axis.legend(loc='upper right', fontsize=9)
        pf.panel_label(axis, material + ', full bandwidth', index=0, loc='lower right'); save(figure, 'gap_fbw_T')
