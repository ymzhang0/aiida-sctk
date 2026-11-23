#!/usr/bin/env python
"""Analytic model of the ALDA spin kernel I(r) of a monatomic crystal and its Fourier transform (auxiliary).

usage: kernel_fit.py PK OUTDIR      PK = SCTKCalculation of the sctk-dump code (as plot_kernel.py)
Model: I(r) = I_flat - sum_R g(|r - R|): a flat interstitial value minus one dip per atom.  Then, for G != 0,
I(G) = -g~(G) / Omega with g~ the 3D Fourier transform of one dip.  Dips fitted to the spherical average of the data:
  exponential  g = D exp(-r/a)        ->  g~(q) = 8 pi D a^3 / (1 + a^2 q^2)^2     (q^-4 tail)
  Gaussian     g = D exp(-r^2/2s^2)   ->  g~(q) = D (2 pi)^(3/2) s^3 exp(-s^2 q^2 / 2)
  two exponentials (core + valence shell)
For comparison the bare Coulomb 1/r transforms to 4 pi / q^2 (q^-2 tail).
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from aiida import orm, load_profile
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figure-style'))
import paperfig as pf
load_profile('sf')
RY, BOHR = 13.605693123, 0.529177210903
node = orm.load_node(int(sys.argv[1])); out = sys.argv[2]
structure = node.inputs.parent_folder.creator.inputs.structure; name = structure.get_formula()
A = np.array(structure.cell) / BOHR; omega = abs(np.linalg.det(A)); B = 2 * np.pi * np.linalg.inv(A).T
with open(os.path.join(node.get_remote_workdir(), 'sctk_xc_kernel.bin'), 'rb') as h:
    n1, n2, n3, n1x, n2x, n3x = np.fromfile(h, dtype=np.int32, count=6)
    rho, d11, d22 = (np.fromfile(h, dtype=np.float64, count=n1x * n2x * n3x).reshape((n1x, n2x, n3x), order='F')[:n1, :n2, :n3] for _ in range(3))
I = -2 * d22 * RY
# distance of every grid point to the nearest atom (atom at the origin; periodic images)
frac = np.stack(np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2, np.arange(n3) / n3, indexing='ij'), -1).reshape(-1, 3)
shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
r = np.min(np.linalg.norm((frac[:, None, :] - shifts[None]) @ A, axis=2), axis=1)
Iv = I.ravel(); iflat = Iv.max()
bins = np.linspace(0, r.max(), 28); c = 0.5 * (bins[1:] + bins[:-1]); which = np.digitize(r, bins) - 1
prof = np.array([Iv[which == i].mean() if np.any(which == i) else np.nan for i in range(len(c))]); ok = ~np.isnan(prof)
dip = iflat - prof
fgau = lambda x, D, s: D * np.exp(-x ** 2 / (2 * s ** 2))
fstep = lambda x, D, r0, w: D / (1 + np.exp((x - r0) / w))            # soft sphere: depth D, radius r0, edge width w
pg, _ = curve_fit(fgau, c[ok], dip[ok], p0=[dip[ok][0], 1.0]); ps, _ = curve_fit(fstep, c[ok], dip[ok], p0=[dip[ok][0], 1.5, 0.2])
rr = np.linspace(1e-4, 12.0, 6000)
def radial_ft(g, q):   # 3D Fourier transform of a spherical function: 4 pi int g(r) sin(qr)/(qr) r^2 dr
    return np.array([4 * np.pi * np.trapezoid(g(rr) * np.sinc(x * rr / np.pi) * rr ** 2, rr) for x in q])
sphere = lambda q, D, R: 4 * np.pi * D * (np.sin(q * R) - q * R * np.cos(q * R)) / q ** 3      # sharp sphere, envelope ~ q^-2
IG = np.fft.fftn(I) / I.size / omega                                   # eV: (1/Omega^2) int I(r) exp(-iG.r) d3r
m = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1 / n) for n in (n1, n2, n3)], indexing='ij'), -1).reshape(-1, 3)
gn = np.linalg.norm(m @ B, axis=1); keep = (gn > 1e-8) & (gn < 6.0)
print(f'{name}: I_flat = {iflat:.1f} eV bohr^3; soft sphere depth D = {ps[0]:.1f} eV bohr^3, radius r0 = {ps[1]:.3f} bohr, edge width w = {ps[2]:.3f} bohr; '
      f'Gaussian D = {pg[0]:.1f}, sigma = {pg[1]:.3f} bohr; nearest-neighbour distance {min(np.linalg.norm(np.array(v) @ A) for v in shifts if any(v)):.3f} bohr')
for lab, res in (('soft sphere', dip[ok] - fstep(c[ok], *ps)), ('gauss', dip[ok] - fgau(c[ok], *pg))):
    print(f'   rms residual of the radial fit, {lab}: {np.sqrt(np.mean(res ** 2)):.2f} eV bohr^3')
q = np.geomspace(0.8, 6.0, 400)
ms_ = np.abs(radial_ft(lambda x: fstep(x, *ps), q)) / omega ** 2; mg_ = np.abs(radial_ft(lambda x: fgau(x, *pg), q)) / omega ** 2
with pf.style('single'), plt.rc_context({'svg.fonttype': 'path'}):
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={'wspace': 0.27})
    for axis in axes: pf.style_axis(axis)
    x = np.linspace(0, r.max(), 200)
    axes[0].plot(r[::7], Iv[::7], color='0.7', linestyle='none', marker='o', markersize=2, label='grid points')
    axes[0].plot(c[ok], prof[ok], color='black', marker='o', linestyle='none', label='spherical average')
    axes[0].plot(x, iflat - fstep(x, *ps), color=pf.color('blue'), linestyle='-', label='soft sphere (Fermi step)')
    axes[0].plot(x, iflat - fgau(x, *pg), color=pf.color('orange'), linestyle='--', label='Gaussian dip')
    axes[0].set_xlabel('Distance to the nearest atom (bohr)'); axes[0].set_ylabel(r'$I(\mathbf{r})$ (eV bohr$^{3}$)'); axes[0].legend(loc='lower right', fontsize=10)
    pf.panel_label(axes[0], name, index=0, loc='upper left')
    axes[1].plot(gn[keep], np.abs(IG.ravel()[keep]), color='black', linestyle='none', marker='o', markersize=3.5, label=r'$|I(\mathbf{G})|$, FFT of the data')
    axes[1].plot(q, ms_, color=pf.color('blue'), linestyle='-', label='soft sphere')
    axes[1].plot(q, mg_, color=pf.color('orange'), linestyle='--', label='Gaussian dip')
    axes[1].plot(q, 4 * np.pi * ps[0] * ps[1] / q ** 2 / omega ** 2, color=pf.color('green'), linestyle=':', label=r'sharp-sphere envelope $\propto G^{-2}$')
    axes[1].plot(q, 8 * np.pi / q ** 2 / omega * RY, color='0.45', linestyle='-.', linewidth=0.9, label=r'bare Coulomb $4\pi e^2/\Omega G^2$')
    axes[1].set_yscale('log'); axes[1].set_xlabel(r'$|\mathbf{G}|$ (bohr$^{-1}$)'); axes[1].set_ylabel('Matrix element (eV)'); axes[1].set_ylim(1e-4, 30)
    axes[1].legend(loc='upper right', fontsize=9); pf.panel_label(axes[1], name, index=1, loc='lower left')
    base = os.path.join(out, f'{name}_kernel_analytic'); figure.savefig(base + '.svg', bbox_inches='tight', pad_inches=0.02); print('wrote', pf.save(figure, base + '.pdf'))
# log-log slope of the data envelope
sel = keep & (gn > 2.0); sl = np.polyfit(np.log(gn[sel]), np.log(np.abs(IG.ravel()[sel]) + 1e-12), 1)[0]; print(f'log-log slope of |I(G)| for G > 2 bohr^-1: {sl:.2f}')
