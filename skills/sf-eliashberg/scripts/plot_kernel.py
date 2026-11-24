#!/usr/bin/env python
"""Figures of the local xc kernels dumped by the SCTK_dump build (auxiliary; not in the AiiDA provenance).

usage: plot_kernel.py PK OUTDIR      PK = SCTKCalculation run with code sctk-dump (file sctk_xc_kernel.bin in its work dir)
I(r) = -2 dmuxc(r,2,2) (spin channel), f(r) = dmuxc(r,1,1) (density channel, LDA part), both in eV bohr^3.
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from aiida import orm, load_profile
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figure-style'))
import paperfig as pf
load_profile('sf')
RY, BOHR = 13.605693123, 0.529177210903
node = orm.load_node(int(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
structure = node.inputs.parent_folder.creator.inputs.structure; name = {'B2Mg': 'MgB2', 'RuSbZr': 'ZrRuSb'}.get(structure.get_formula(), structure.get_formula())
A = np.array(structure.cell) / BOHR; omega = abs(np.linalg.det(A)); B = 2 * np.pi * np.linalg.inv(A).T
with open(os.path.join(node.get_remote_workdir(), 'sctk_xc_kernel.bin'), 'rb') as h:
    n1, n2, n3, n1x, n2x, n3x = np.fromfile(h, dtype=np.int32, count=6)
    rho, d11, d22 = (np.fromfile(h, dtype=np.float64, count=n1x * n2x * n3x).reshape((n1x, n2x, n3x), order='F')[:n1, :n2, :n3] for _ in range(3))
I = -2 * d22 * RY; f = d11 * RY                                       # eV bohr^3
print(f'{name}: grid {n1}x{n2}x{n3}, Omega = {omega:.3f} bohr^3, integral of rho = {rho.mean() * omega:.3f}')
print(f'I(r): min {I.min():.2f}, mean {I.mean():.2f}, max {I.max():.2f} eV bohr^3; I_00 = {I.mean() / omega:.4f} eV')
print(f'f(r): min {f.min():.2f}, mean {f.mean():.2f}, max {f.max():.2f} eV bohr^3; f_00 = {f.mean() / omega:.4f} eV')
# exchange-only LDA: e_x = C n^(4/3) [Ha]; d2e_x/dn2 = d2e_x/dm2 = (4C/9) n^(-2/3)  ->  f_x = (4C/9) n^(-2/3), I_x = -2 f_x
C = -0.75 * (3 / np.pi) ** (1 / 3)
fx = lambda n: (4 * C / 9) * n ** (-2 / 3) * 2 * RY                   # Ha -> eV
def save(figure, tag):
    base = os.path.join(out, f'{name}_{tag}'); figure.savefig(base + '.svg', bbox_inches='tight', pad_inches=0.02)
    print('wrote', pf.save(figure, base + '.pdf'))

with pf.style('single'), plt.rc_context({'svg.fonttype': 'path'}):
    # figure 1: real space.  (a) along the line joining nearest neighbours, (b) against the local density
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={'wspace': 0.27})
    for axis in axes: pf.style_axis(axis)
    if n1 == n2 == n3:
        idx = np.arange(n1 + 1) % n1; line = lambda a: a[idx, idx, idx]
        dist = np.linalg.norm(A.sum(axis=0)) * np.arange(n1 + 1) / n1 * BOHR         # Angstrom along a1 + a2 + a3
        xlabel = r'Distance along the nearest-neighbour line ($\mathrm{\AA}$)'
    else:                    # non-cubic grid: straight line from the first atom through the second (trilinear interpolation)
        from scipy.ndimage import map_coordinates
        sites = structure.sites; frac = np.linalg.solve((A * BOHR).T, np.array(sites[1].position) - np.array(sites[0].position))
        t = np.linspace(0, 2, 241); pts = (np.outer(t, frac) * np.array([n1, n2, n3])).T
        line = lambda a: map_coordinates(a, pts, order=1, mode='grid-wrap')
        dist = t * np.linalg.norm(np.array(sites[1].position) - np.array(sites[0].position))
        xlabel = rf'Distance along the {sites[0].kind_name}$-${sites[1].kind_name} line ($\mathrm{{\AA}}$)'
    axes[0].plot(dist, line(I), color=pf.color('blue'), linestyle='-', label=r'$I(\mathbf{r})$ (spin)')
    axes[0].plot(dist, -line(f), color=pf.color('orange'), linestyle='--', label=r'$-f(\mathbf{r})$ (density)')
    axes[0].set_yscale('log'); axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(r'Kernel (eV bohr$^{3}$)'); axes[0].legend(loc='lower center'); pf.panel_label(axes[0], name, index=0, loc='upper left')
    step = max(1, rho.size // 4000); order = np.argsort(rho.ravel())[::step]
    nn = rho.ravel()[order]; grid = np.geomspace(nn[nn > 0].min(), nn.max(), 200)
    axes[1].plot(nn, I.ravel()[order], color=pf.color('blue'), linestyle='none', marker='o', markersize=2.5, rasterized=True)
    axes[1].plot(nn, -f.ravel()[order], color=pf.color('orange'), linestyle='none', marker='s', markersize=2.5, rasterized=True)
    axes[1].plot(grid, -2 * fx(grid), color='black', linestyle='--', linewidth=0.9)
    axes[1].plot(grid, -fx(grid), color='black', linestyle=':', linewidth=0.9)
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_xlabel(r'Local density $n(\mathbf{r})$ (bohr$^{-3}$)')
    axes[1].set_ylabel(r'Kernel (eV bohr$^{3}$)')
    axes[1].legend(handles=[Line2D([], [], color=pf.color('blue'), marker='o', linestyle='none', label=r'$I$'),
                            Line2D([], [], color=pf.color('orange'), marker='s', linestyle='none', label=r'$-f$'),
                            Line2D([], [], color='black', linestyle='--', linewidth=0.9, label=r'exchange only, $I_x$'),
                            Line2D([], [], color='black', linestyle=':', linewidth=0.9, label=r'exchange only, $-f_x$')], loc='upper right', fontsize=10)
    pf.panel_label(axes[1], name, index=1, loc='lower left')
    save(figure, 'kernel_real_space')

    # figure 2: reciprocal space.  (a) Fourier components against |G| with the bare Coulomb, (b) |I_GG'| for the shortest G
    IG = np.fft.fftn(I) / I.size / omega; fG = np.fft.fftn(f) / f.size / omega     # eV: matrix element X_{G G'} = X(G - G')
    m = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1 / n) for n in (n1, n2, n3)], indexing='ij'), -1).reshape(-1, 3)
    gnorm = np.linalg.norm(m @ B, axis=1); keep = (gnorm > 1e-8) & (gnorm < 6.0)
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={'wspace': 0.3})
    pf.style_axis(axes[0])
    gg = np.linspace(0.5, 6.0, 300)
    axes[0].plot(gg, 8 * np.pi / gg ** 2 / omega * RY, color='black', linestyle='-', label=r'bare Coulomb $v(G)$')
    axes[0].plot(gnorm[keep], np.abs(IG.ravel()[keep]), color=pf.color('blue'), linestyle='none', marker='o', markersize=3.5, label=r'$|I(\mathbf{G})|$')
    axes[0].plot(gnorm[keep], np.abs(fG.ravel()[keep]), color=pf.color('orange'), linestyle='none', marker='s', markersize=3.5, markerfacecolor='white', label=r'$|f(\mathbf{G})|$')
    pf.reference_line(axes[0], I.mean() / omega); pf.reference_line(axes[0], -f.mean() / omega, kind='zero')
    axes[0].plot([], [], color='black', linestyle='--', linewidth=0.9, label=r'$I_{00}$ (diagonal)'); axes[0].plot([], [], color='black', linestyle=':', linewidth=0.7, label=r'$|f_{00}|$ (diagonal)')
    axes[0].set_yscale('log'); axes[0].set_xlabel(r'$|\mathbf{G}|$ (bohr$^{-1}$)'); axes[0].set_ylabel('Matrix element (eV)')
    axes[0].set_ylim(1e-4, 300); axes[0].legend(loc='upper right', fontsize=9, ncol=2, columnspacing=0.8); pf.panel_label(axes[0], name, index=0, loc='lower left')
    sel = np.argsort(gnorm, kind='stable')[:27]; ms = m[sel].astype(int)
    M = np.array([[abs(IG[tuple((a - b) % np.array([n1, n2, n3]))]) for b in ms] for a in ms])
    # vector cells, not an embedded bitmap: the web document drops bitmaps embedded in an SVG (the panel came out empty)
    image = axes[1].pcolormesh(np.arange(28) - 0.5, np.arange(28) - 0.5, np.log10(np.maximum(M, 1e-6)), cmap='magma', rasterized=False, linewidth=0); axes[1].set_aspect('equal')
    cb = figure.colorbar(image, ax=axes[1]); cb.solids.set_rasterized(False)   # matplotlib rasterises long colour bars; the web document drops bitmaps
    cb.set_label(r'$\log_{10}|I_{\mathbf{G}\mathbf{G}^\prime}|$ (eV)')
    axes[1].set_xlabel(r"$\mathbf{G}'$ index (27 shortest)"); axes[1].set_ylabel(r'$\mathbf{G}$ index')
    axes[1].text(0.04, 0.93, f'(b) {name}', transform=axes[1].transAxes, color='white', va='top')
    save(figure, 'kernel_reciprocal_space')
    cross = gg[np.argmin(np.abs(8 * np.pi / gg ** 2 / omega * RY + f.mean() / omega))]
    print(f'v(G) = |f_00| at G = {cross:.2f} bohr^-1; largest off-diagonal |I_G| = {np.abs(IG.ravel()[keep]).max():.4f} eV, |f_G| = {np.abs(fG.ravel()[keep]).max():.4f} eV')
