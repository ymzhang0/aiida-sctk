#!/usr/bin/env python
"""Convergence figures for the SCTK spin-fluctuation runs (auxiliary; not part of the AiiDA provenance).

usage: plot_conv.py MATERIAL OUTDIR      (house style = skill paper-figure-style; two-panel figures: left lambda_max, right lambda_sf)
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from aiida import orm
import qavg
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figure-style'))
import paperfig as pf
from aiida_sctk.parsers.raw import parse_head_file

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
MARK = ['o', 's', '^', 'D', 'v', 'P']
INK, MUTED, GRID, SURF = '#1a1a19', '#6b6a63', '#e4e3de', '#fcfcfb'
RUNS = {  # parameter value -> pk; one parameter varied at a time from the baseline
    'V': dict(ecutfock={20: 33, 30: 511, 40: 541, 50: 572, 60: 2027, 70: 2057}, nbnd={20: 33, 30: 603, 40: 633},
              k={16: 33, 24: 131}, q={4: 131, 6: 162, 8: 3070, 10: 2483, 12: 1997},
              k_q8={16: 3039, 32: 3070}),   # only k grids commensurate with the q grid (method page 5.5)
    'Nb': dict(ecutfock={20: 481, 30: 726, 40: 753, 50: 787}, nbnd={20: 481, 30: 808, 40: 838, 50: 2088, 60: 2119},
               k={16: 481, 24: 225}, q={4: 225, 6: 256}),
    'Pb': dict(ecutfock={20: 3948, 30: 4009, 40: 4039, 50: 4070, 60: 4097}, nbnd={20: 3948, 30: 4124, 40: 4154, 50: 6041, 60: 6071},
               k={16: 3948, 24: 4220}, q={4: 4220, 6: 4268}),
    'MgB2': dict(ecutfock={20: 3978, 30: 4349, 40: 4370, 50: 4393, 60: 4432}, nbnd={20: 3978, 30: 4474, 40: 4495, 50: 6101, 60: 6132, 80: 6162},
                 k={16: 3978, 24: 4553}, q={4: 4553, 6: 4608}),
    # half-Heuslers: baseline q 6^3 unshifted, k 18^3, 40 bands, ecutfock 30; the q series pairs each q grid with a commensurate k grid
    'HfRuSb': dict(ecutfock={20: 10036, 30: 8950, 40: 10067, 50: 10099}, nbnd={40: 8950, 60: 13725, 80: 10161, 100: 14474},
                   k={12: 6825, 18: 8950, 24: 6856}, q={4: 6884, 6: 8950, 8: 10420}),
    # VN (8-atom cell): baseline q 4^3 unshifted, k 12^3, 56 bands, ecutfock 30, nci 12 + ci_scale 0.3
    'VN': dict(ecutfock={20: 12058, 30: 12027, 40: 12089}, nbnd={56: 12027, 80: 12122, 120: 12153, 160: 14599},
               k={12: 12027, 16: 12181}, q={4: 12027, 6: 12205}),
    'ZrRuSb': dict(ecutfock={20: 10226, 30: 8981, 40: 10257, 50: 10279}, nbnd={40: 8981, 60: 10319, 80: 13756, 100: 14536},
                   k={12: 7165, 18: 8981, 24: 7196}, q={4: 7235, 6: 8981, 8: 10451}),
}
# screened Coulomb kernel with laddxc = 1, 2 (launch_sf.py --laddxc, tags xc1_*, xc2_*), same scan as laddxc = 0 where it exists
def _xc(material, lx, tags):
    g = orm.load_group(f'sf/sctk/{material}'); lab = {n.label.split(':')[1]: n.pk for n in g.nodes if n.label.split(':')[1].startswith(f'xc{lx}_')}
    return {k: lab[f'xc{lx}_{t}'] for k, t in tags.items()}
XC = {'V': {0: dict(ecutfock={20: 33, 30: 511, 40: 541, 50: 572, 60: 2027}, nbnd={20: 33, 30: 603, 40: 633}, k={16: 33, 24: 131}, q={4: 131, 6: 162, 8: 3070})}}
for _lx in (1, 2):
    XC['V'][_lx] = dict(ecutfock=_xc('V', _lx, {20: 'base', 30: 'ecutfock30', 40: 'ecutfock40', 50: 'ecutfock50', 60: 'ecutfock60'}),
                        nbnd=_xc('V', _lx, {20: 'base', 30: 'nb30', 40: 'nb40'}), k=_xc('V', _lx, {16: 'base', 24: 'nk24', 32: 'nk32'}),
                        q=_xc('V', _lx, {4: 'nk24', 6: 'q6s_k24', 8: 'q8s_k32'}))
import os
material, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
R = RUNS[material]
cache = {}
def get(pk):
    if pk not in cache:
        node, nu, lam, eig, w, frac = qavg.analyse(pk)
        cache[pk] = dict(nu=nu, lam=lam, eig=np.array(eig), node=node)
    return cache[pk]
def lam_at(pk): d = get(pk); return [np.interp(x, d['nu'], d['lam']) for x in (0.0, 0.1, 1.0)]
def muc_at(pk):
    """q-averaged Coulomb kernel mu_C(i nu) at 0, 0.1, 1 eV (same star weights as lambda_sf); None without a merged kernel."""
    node = get(pk)['node']
    if 'kernel' not in node.outputs: return None
    _, nu, _, _, w, _ = qavg.analyse(pk); return [np.interp(x, nu, w @ node.outputs.kernel.get_array('mu_coulomb')) for x in (0.0, 0.1, 1.0)]
def qlabels(pk):
    n = get(pk)['node']
    par = n.outputs.output_parameters if 'output_parameters' in n.outputs else \
        [c for c in n.called_descendants if c.process_label == 'SCTKCalculation'][0].outputs.output_parameters
    q = np.abs(np.array(par.get_dict()['qpoints_cartesian'])); alat = np.linalg.norm(n.inputs.structure.cell[0])
    cell = np.array(n.inputs.structure.cell)
    a = {'VN': np.linalg.norm(cell[0]), 'Pb': 2 * np.abs(cell[0]).max(), 'HfRuSb': 2 * np.abs(cell[0]).max(), 'ZrRuSb': 2 * np.abs(cell[0]).max(), 'MgB2': np.linalg.norm(cell[0])}.get(material, abs(cell[0][0]) * 2)   # conventional a
    q = np.sort(q * a / alat, axis=1)                                   # units of 2 pi / a
    frac = {0.0: '0', 0.25: '¼', 0.5: '½', 0.75: '¾', 1.0: '1'}
    return ['(' + ', '.join(frac.get(round(x, 3), f'{x:.3g}') for x in row) + ')' for row in q], np.linalg.norm(q, axis=1)

# one mapping per physical quantity, shared by every figure (colour + marker + line style)
Q_STYLE = [('blue', 'o', '-'), ('orange', 's', '--'), ('green', 'D', '-'), ('purple', '^', '--'), ('yellow', 'v', '-'), ('sky', 'P', '--'),
           ('black', 'X', '-'), ('blue', 'o', ':'), ('orange', 's', ':'), ('green', 'D', ':')]   # > 7 q points: colour repeats with a dotted line
FREQ = [(r'$\nu = 0$', 'black', 'o', '-'), (r'$\nu = 0.1$ eV', 'blue', 's', '--'), (r'$\nu = 1$ eV', 'orange', 'D', ':')]
GRID_STYLE = {4: ('black', 'o'), 6: ('blue', 's'), 8: ('orange', 'D'), 10: ('green', '^'), 12: ('purple', 'v')}
ADOPTED = {'V': dict(ecutfock=60, nbnd=20, k=32, q=8), 'Nb': dict(ecutfock=30, nbnd=40, k=24, q=6),
           'Pb': dict(ecutfock=20, nbnd=40, k=24, q=6), 'MgB2': dict(ecutfock=20, nbnd=40, k=24, q=6)}.get(material, {})
XL = {'ecutfock': r'$E_{\mathrm{cut}}^{\mathrm{Fock}}$ (Ry)', 'nbnd': 'Number of bands', 'k': r'$N_k$ ($N_k^3$ $k$ grid)'}
YMAX, YSF = r'$\lambda^{\max}$', r'$\lambda^{\mathrm{sf}}(i\nu)$'
RY = 13.605693123
# FSM runs of the user (AiiDA export): total energy [Ry] against the fixed moment [mu_B per cell]; production kel run
FSM = {'V': dict(m=[0.0, 0.05, 0.10], e=[-147.61686470322573, -147.6168538209209, -147.6168201615626], n0=0.9408, prod=8821),
       'Nb': dict(m=[0.0, 0.05, 0.10], e=[-124.57142104849898, -124.57139389195527, -124.57130972546042], n0=0.7961, prod=8852),
       'Pb': dict(m=[0.0, 0.05, 0.10], e=[-140.54779887719945, -140.54766255932995, -140.54722970005207], n0=0.2776, prod=8874),
       'MgB2': dict(m=[0.0, 0.05, 0.10], e=[-130.5836759556612, -130.58358407937905, -130.58329940041673], n0=0.3755, prod=8916)}
# FSM runs of group sf/fsm/<material> (launch_fsm.py: k 36^3, MgB2 32x32x24, cold smearing 0.005 Ry); n0 = DOS per spin at E_F
# of the m = 0 run with the same smearing (fsm_table.py).  The earlier figure used the user's runs (cold 0.02 Ry, coarser k)
# together with the tetrahedron DOS of the kel run, which is not consistent with the smeared FSM curvature.
labels, _ = qlabels(list(R['k'].values())[0])
labels = [l for l, n in zip(*qlabels(list(R['k'].values())[0])) if n > 1e-8] if len(labels) == len(get(list(R['k'].values())[0])['eig']) + 1 else labels   # unshifted grid: Gamma listed, not computed
labels = [l.replace('¼', '1/4').replace('½', '1/2').replace('¾', '3/4') for l in labels]

def pair():
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={'wspace': 0.27})
    for axis in axes: pf.style_axis(axis)
    return figure, axes
def headroom(axis, fraction):
    low, high = axis.get_ylim(); axis.set_ylim(low, high + fraction * (high - low))
def save(figure, name):
    base = os.path.join(out, f'{material}_{name}')
    figure.savefig(base + '.svg', bbox_inches='tight', pad_inches=0.02)      # vector copy for the web document
    print('wrote', pf.save(figure, base + '.pdf'))

def draw_lmax(axis, key):
    xs = sorted(R[key]); E = np.array([get(R[key][x])['eig'] for x in xs])
    if E.shape[1] > len(Q_STYLE):      # too many q points for one entry each: all in grey, the largest one highlighted
        for iq in range(E.shape[1]): axis.plot(xs, E[:, iq], color='0.6', marker='o', markersize=3, linewidth=0.9)
        top = int(np.argmax(E[0])); axis.plot(xs, E[:, top], color=pf.color('blue'), marker='o')
        axis.set_xlabel(XL[key]); axis.set_xticks(xs); axis.set_ylabel(YMAX); axis.yaxis.set_major_locator(MaxNLocator(nbins=5)); headroom(axis, 0.3)
        axis.legend(handles=[Line2D([], [], color='0.6', marker='o', markersize=3, label=f'{E.shape[1]} $q$ points'),
                             Line2D([], [], color=pf.color('blue'), marker='o', label='largest: ' + labels[top])], ncol=2, loc='upper center', fontsize=10)
        return
    for iq in range(E.shape[1]):
        c, m, ls = Q_STYLE[iq]
        axis.plot(xs, E[:, iq], color=pf.color(c), marker=m, linestyle=ls, label=labels[iq])
    axis.set_xlabel(XL[key]); axis.set_xticks(xs); axis.set_ylabel(YMAX); axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    headroom(axis, 0.5 if E.shape[1] <= 6 else 1.25); axis.legend(ncol=3, loc='upper center', fontsize=10 if E.shape[1] <= 6 else 7.6, columnspacing=1.0 if E.shape[1] <= 6 else 0.6, handlelength=1.8 if E.shape[1] <= 6 else 1.4, handletextpad=0.4, title=r'$\mathbf{q}$ ($2\pi/a$)', title_fontsize=10)

def draw_lsf(axis, key):
    xs = sorted(R[key]); L = np.array([lam_at(R[key][x]) for x in xs]); ticks = set(xs)
    for i, (name, c, m, ls) in enumerate(FREQ):
        axis.plot(xs, L[:, i], color=pf.color(c), marker=m, linestyle=ls)
    handles = [Line2D([], [], color=pf.color(c), marker=m, linestyle=ls, label=name) for name, c, m, ls in FREQ]
    star = None
    if key == 'k' and 'k_q8' in R:
        xs8 = sorted(R['k_q8']); L8 = np.array([lam_at(R['k_q8'][x]) for x in xs8]); ticks |= set(xs8)
        for i, (name, c, m, ls) in enumerate(FREQ):
            axis.plot(xs8, L8[:, i], color=pf.color(c), marker=m, linestyle=ls, markerfacecolor='white')
        handles += [Line2D([], [], color='0.45', marker='o', linestyle='none', label=r'$q$ $4^3$'),
                    Line2D([], [], color='0.45', marker='o', linestyle='none', markerfacecolor='white', label=r'$q$ $8^3$')]
        if ADOPTED.get('k') in xs8: star = (ADOPTED['k'], L8[xs8.index(ADOPTED['k']), 0])
    elif ADOPTED.get(key) in xs:
        star = (ADOPTED[key], L[xs.index(ADOPTED[key]), 0])
    if star:
        pf.highlight(axis, *star); handles.append(Line2D([], [], color='black', marker='*', markersize=10, linestyle='none', label='adopted'))
    axis.set_xlabel(XL[key]); axis.set_xticks(sorted(ticks)); axis.set_ylabel(YSF); axis.set_ylim(bottom=0)
    headroom(axis, 0.42); axis.legend(handles=handles, ncol=3, loc='upper center', fontsize=10, columnspacing=1.0, handlelength=1.8)

def kel_static(pk):
    """Per irreducible q of work chain pk: |q| [2 pi/a], static chi0 per spin [1/eV per cell], (I_xc Pi_0)_00, lambda_max."""
    node = orm.load_node(pk); rows = {}
    for calc in node.called_descendants:
        if calc.process_label != 'SCTKCalculation' or 'polarizability' not in calc.outputs: continue
        par = calc.outputs.output_parameters.get_dict()
        # plain readers of the plugin (aiida_sctk.parsers.raw); static = first frequency, last column = 1/eV per cell
        for block in parse_head_file(calc.outputs.polarizability.get_content())['qpoints']:
            rows[block['index']] = dict(chi=-block['values'][0][-1].real / 2)
        for block in parse_head_file(calc.outputs.spin_susceptibility.get_content())['qpoints']:
            if block['channel'] == 2: rows[block['index']]['chis'] = -block['values'][0][-1].real / 2
        for i, x, e in zip(par['computed_qpoint_indices'], par['ixc_pi0_head'], par['eig_max']):
            rows[i].update(ixc=x, eig=e)
    _, qn = qlabels(pk); order = sorted(rows)
    return (np.array([qn[i - 1] for i in order]), *(np.array([rows[i][k] for i in order]) for k in ('chi', 'ixc', 'eig', 'chis'))), get(pk)['node']

with pf.style('single'), plt.rc_context({'svg.fonttype': 'path'}):
    if material in FSM:      # (a) fixed-spin-moment fit, (b) Stoner I: FSM against the ALDA kernel of the kel run
        f = FSM[material]; m = np.array(f['m']); de = (np.array(f['e']) - f['e'][0]) * RY * 1e3          # meV
        a, b4 = np.linalg.solve(np.array([[m[1] ** 2, m[1] ** 4], [m[2] ** 2, m[2] ** 4]]), de[1:])           # dE = a m^2 + b m^4 (meV): a is the m -> 0 curvature
        (qn, chi, ixc, eig, chis), node = kel_static(f['prod'])
        n0 = f['n0']                                                                                     # states / eV / spin, same smearing as the FSM runs
        i_fsm = 1 / n0 - 4 * a * 1e-3
        figure, axes = pair()
        mm = np.linspace(0, m.max() * 1.08, 100)
        axes[0].plot(mm, a * mm ** 2 + b4 * mm ** 4, color=pf.color('blue'), linestyle='--', label=r'fit $\Delta E = a\,m^2 + b\,m^4$')
        axes[0].plot(m, de, color=pf.color('black'), marker='o', linestyle='none', label='FSM total energy')
        axes[0].set_xlabel(r'$m$ ($\mu_B$ per cell)'); axes[0].set_ylabel(r'$\Delta E$ (meV)'); axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))
        axes[0].text(0.05, 0.60, f'$a$ = {a:.1f} meV/$\\mu_B^2$\n$N(0)$ = {n0:.3f} eV$^{{-1}}$ per spin\n$I = 1/N(0) - 4a$ = {i_fsm:.3f} eV\n$I\\,N(0)$ = {i_fsm * n0:.3f}',
                     transform=axes[0].transAxes, ha='left', va='top', fontsize=11)
        pf.panel_label(axes[0], material, index=0)
        ihead = 1 / chi - 1 / chis                               # method page 3.4: I = 1/chi0_00 - 1/chis_00 (per spin)
        axes[1].plot(qn, ihead, color=pf.color('blue'), marker='o', linestyle='none', label=r'ALDA: $1/\chi^0_{00} - 1/\chi^{s}_{00}$')
        pf.reference_line(axes[1], i_fsm); axes[1].plot([], [], color='black', linestyle='--', linewidth=0.9, label='FSM $I$')
        axes[1].set_xlabel(r'$|\mathbf{q}|$ ($2\pi/a$)'); axes[1].set_ylabel(r'$I$ (eV)'); axes[1].set_ylim(bottom=0); headroom(axes[1], 0.35)
        axes[1].legend(loc='upper center', fontsize=11, ncol=2); pf.panel_label(axes[1], material, index=1, loc='lower right')
        save(figure, 'stoner_fsm_alda')
        o = np.argsort(qn)
        print(f'FSM: a = {a:.3f} meV/muB^2, N0 = {n0:.4f} /eV/spin, I = {i_fsm:.4f} eV; ALDA head I: smallest q {qn[o][0]:.3f} -> {ihead[o][0]:.4f} eV, range {ihead.min():.3f}-{ihead.max():.3f}; chi0 {chi[o][0]:.4f}, chis {chis[o][0]:.4f} /eV/spin; first 4:', np.round(ihead[o][:4], 4), np.round(qn[o][:4], 3))

    for key in ('ecutfock', 'nbnd', 'k'):
        figure, axes = pair(); draw_lmax(axes[0], key); draw_lsf(axes[1], key)
        pf.panel_label(axes[0], material, index=0, loc='lower right'); pf.panel_label(axes[1], material, index=1, loc='lower right')
        save(figure, f'conv_{key}')

    grids = sorted(R['q']); figure, axes = pair()
    for n in grids:
        lab, qn = qlabels(R['q'][n]); d = get(R['q'][n]); c, mk = GRID_STYLE[n]
        if len(qn) == len(d['eig']) + 1: qn = qn[qn > 1e-8]      # unshifted grid: Gamma is listed but not computed
        if len(qn) == len(d['eig']):
            axes[0].plot(qn, d['eig'], linestyle='none', marker=mk, markersize=4.5, color=pf.color(c), label=rf'${n}^3$')
    axes[0].set_xlabel(r'$|\mathbf{q}|$ ($2\pi/a$)'); axes[0].set_ylabel(YMAX); axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    headroom(axes[0], 0.22); axes[0].legend(ncol=len(grids), loc='upper center', fontsize=10, columnspacing=0.8, handletextpad=0.2)
    pts = np.array([axes[0].transAxes.inverted().transform(axes[0].transData.transform(p)) for line in axes[0].lines for p in line.get_xydata()])
    corners = {'lower right': (1, 0), 'lower left': (0, 0), 'upper left': (0, 0.8)}      # the legend occupies the top centre
    free = min(corners, key=lambda c: np.sum((np.abs(pts[:, 0] - corners[c][0]) < 0.45) & (np.abs(pts[:, 1] - corners[c][1]) < 0.16)))
    pf.panel_label(axes[0], material + r', $q$ grids', index=0, loc=free if free != 'upper left' else 'lower left')
    L = np.array([lam_at(R['q'][n]) for n in grids])
    for i, (name, c, mk, ls) in enumerate(FREQ):
        axes[1].plot(grids, L[:, i], color=pf.color(c), marker=mk, linestyle=ls, label=name)
    if ADOPTED.get('q') in grids:
        pf.highlight(axes[1], ADOPTED['q'], L[grids.index(ADOPTED['q']), 0], label='adopted')
    axes[1].set_xlabel(r'$N_q$ ($N_q^3$ $q$ grid)'); axes[1].set_xticks(grids); axes[1].set_ylabel(YSF); axes[1].set_ylim(bottom=0)
    headroom(axes[1], 0.3); axes[1].legend(ncol=2, loc='upper center', fontsize=10, columnspacing=1.0)
    pf.panel_label(axes[1], material, index=1, loc='lower right')
    save(figure, 'conv_q')

    # screened Coulomb kernel: same scans, q-averaged mu_C(i nu); two panels per figure (left: cutoff / bands, right: k / q grid)
    YC = r'$\mu^{C}(i\nu)$'
    def draw_muc(axis, runs, xlabel, star=None):
        xs = [x for x in sorted(runs) if muc_at(runs[x]) is not None]; L = np.array([muc_at(runs[x]) for x in xs])
        for i, (name, c, m, ls) in enumerate(FREQ):
            axis.plot(xs, L[:, i], color=pf.color(c), marker=m, linestyle=ls, label=name)
        if star in xs: pf.highlight(axis, star, L[xs.index(star), 0], label='adopted')
        axis.set_xlabel(xlabel); axis.set_xticks(xs); axis.set_ylabel(YC); axis.set_ylim(bottom=0); headroom(axis, 0.32)
        axis.legend(ncol=2, loc='upper center', fontsize=10, columnspacing=1.0)
        print(f'mu_C {xlabel}:', {x: [round(v, 4) for v in row] for x, row in zip(xs, L)})
    if material in XC:       # laddxc = 0 / 1 / 2 in the same panels: colour = laddxc, line style = frequency (0 and 1 eV)
        X = XC[material]; XCS = [(0, 'black', 'o', 'RPA (laddxc 0)'), (1, 'blue', 's', 'laddxc 1'), (2, 'orange', 'D', 'KO (laddxc 2)')]
        def draw_xc(axis, key, xlabel, star=None):
            ticks = set()
            for lx, c, mk, lab in XCS:
                runs = X[lx][key]; xs = sorted(runs); L = np.array([muc_at(runs[x]) for x in xs]); ticks |= set(xs)
                axis.plot(xs, L[:, 0], color=pf.color(c), marker=mk, linestyle='-'); axis.plot(xs, L[:, 2], color=pf.color(c), marker=mk, linestyle=':', markerfacecolor='white')
                print(f'mu_C laddxc {lx} {key}:', {x: [round(float(v), 4) for v in row] for x, row in zip(xs, L)})
            hs = [Line2D([], [], color=pf.color(c), marker=mk, linestyle='-', label=lab) for _, c, mk, lab in XCS]
            hs += [Line2D([], [], color='0.4', linestyle='-', label=r'$\nu = 0$'), Line2D([], [], color='0.4', linestyle=':', label=r'$\nu = 1$ eV')]
            axis.set_xlabel(xlabel); axis.set_xticks(sorted(ticks)); axis.set_ylabel(YC); axis.set_ylim(bottom=0); headroom(axis, 0.45)
            axis.legend(handles=hs, ncol=3, loc='upper center', fontsize=9, columnspacing=0.8, handlelength=1.6)
        figure, axes = pair(); draw_xc(axes[0], 'ecutfock', XL['ecutfock']); draw_xc(axes[1], 'nbnd', XL['nbnd'])
        pf.panel_label(axes[0], material, index=0, loc='lower right'); pf.panel_label(axes[1], material, index=1, loc='lower right'); save(figure, 'conv_coulomb_xc_cutoff_bands')
        figure, axes = pair(); draw_xc(axes[0], 'k', XL['k']); draw_xc(axes[1], 'q', r'$N_q$ ($N_q^3$ $q$ grid)')
        pf.panel_label(axes[0], material, index=0, loc='lower right'); pf.panel_label(axes[1], material, index=1, loc='lower right'); save(figure, 'conv_coulomb_xc_grids')
    figure, axes = pair(); draw_muc(axes[0], R['ecutfock'], XL['ecutfock'], ADOPTED.get('ecutfock')); draw_muc(axes[1], R['nbnd'], XL['nbnd'], ADOPTED.get('nbnd'))
    pf.panel_label(axes[0], material, index=0, loc='lower right'); pf.panel_label(axes[1], material, index=1, loc='lower right'); save(figure, 'conv_coulomb_cutoff_bands')
    figure, axes = pair(); draw_muc(axes[0], R.get('k_q8', R['k']), XL['k'], ADOPTED.get('k')); draw_muc(axes[1], R['q'], r'$N_q$ ($N_q^3$ $q$ grid)', ADOPTED.get('q'))
    pf.panel_label(axes[0], material, index=0, loc='lower right'); pf.panel_label(axes[1], material, index=1, loc='lower right'); save(figure, 'conv_coulomb_grids')
