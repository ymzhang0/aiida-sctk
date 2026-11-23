#!/usr/bin/env python
"""Auxiliary post-processing (not part of the provenance): q-average of the SCTK kernels.

usage: qavg.py PK [PK ...] [--table FILE]
PK = sctk.spin_fluctuation work chain.  Prints the star-weighted q average of EigMax and of
lambda_sf(i nu) = N(0)<K_SF(q, i nu)>_FS at nu = 0, 0.1 and 1 eV (the agreed convergence metric).
--table writes the q-averaged table nu[meV] lambda_sf(i nu) for EPW `fila2f_sf` (first PK only).
"""
import sys
import numpy as np
import spglib
from aiida import orm, load_profile

load_profile('sf')
RY = 13.605693123

def star_weights(structure, mesh, shift, qcart):
    """Weights of the SCTK q points (Cartesian, 2 pi / alat) on the uniform grid, from spglib orbits."""
    atoms = structure.get_ase(); cell = atoms.get_cell().array
    alat = np.linalg.norm(cell[0])
    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, (cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
                                                  is_shift=shift, is_time_reversal=True, symprec=1e-5)
    mesh = np.array(mesh); shift = np.array(shift)
    lookup = {tuple((2 * g + shift) % (2 * mesh)): i for i, g in enumerate(grid)}
    counts = np.bincount(mapping, minlength=len(mapping))
    weights, seen = [], set()
    for q in qcart:
        frac = np.array(q) @ cell.T / alat                       # crystal coordinates
        key = tuple(np.rint(frac * 2 * mesh).astype(int) % (2 * mesh))
        orbit = mapping[lookup[key]]
        assert orbit not in seen, 'two SCTK q points fall in the same orbit'
        seen.add(orbit); weights.append(counts[orbit])
    assert sum(weights) == mesh.prod(), (sum(weights), mesh.prod())
    return np.array(weights, float)

def analyse(pk):
    node = orm.load_node(pk)
    if 'kernel' in node.outputs:
        kernel = node.outputs.kernel; par = node.outputs.output_parameters.get_dict()
    else:  # work chain did not finish cleanly: merge the kernels of its kel calculations
        calcs = [c for c in node.called_descendants if c.process_label == 'SCTKCalculation' and 'kernel' in c.outputs]
        par = calcs[0].outputs.output_parameters.get_dict()
        names = ('qpoint_indices', 'eig_max', 'mu_spin_fluctuation')
        cat = {n: np.concatenate([c.outputs.kernel.get_array(n) for c in calcs]) for n in names}
        order = np.argsort(cat['qpoint_indices'])
        assert len(set(cat['qpoint_indices'])) == len(order) == par['number_of_qpoints'], 'incomplete q coverage'
        class K:
            def get_array(self, n): return calcs[0].outputs.kernel.get_array(n) if n == 'frequencies' else cat[n][order]
        kernel = K()
    w_all = star_weights(node.inputs.structure, par['q_grid'], par['q_shift'], par['qpoints_cartesian'])
    idx = kernel.get_array('qpoint_indices') - 1
    w = w_all[idx] / w_all[idx].sum()
    nu = kernel.get_array('frequencies') * RY                      # eV
    lam = w @ kernel.get_array('mu_spin_fluctuation'); eig = kernel.get_array('eig_max')
    return node, nu, lam, eig, w, w_all[idx].sum() / w_all.sum()

if __name__ == '__main__':
    args = sys.argv[1:]; table = None
    if '--table' in args:
        i = args.index('--table'); table = args[i + 1]; del args[i:i + 2]
    print(f"{'pk':>6} {'label':34} {'nq':>3} {'lam(0)':>8} {'lam(0.1eV)':>10} {'lam(1eV)':>9} {'EigMax avg':>10} {'max':>7} {'BZ frac':>7}")
    ctable = None
    if '--ctable' in args:      # same for the Coulomb kernel mu_C(i nu), EPW `fila2f_c`
        i = args.index('--ctable'); ctable = args[i + 1]; del args[i:i + 2]
    for n, pk in enumerate(args):
        node, nu, lam, eig, w, frac = analyse(int(pk))
        v = np.interp([0.0, 0.1, 1.0], nu, lam)
        print(f"{node.pk:>6} {node.label[:34]:34} {len(eig):>3} {v[0]:8.4f} {v[1]:10.4f} {v[2]:9.4f} {w @ eig:10.4f} {eig.max():7.4f} {frac:7.3f}")
        if table and n == 0:
            with open(table, 'w') as handle:
                handle.write(f"# imag-axis table: nu[meV]  lambda_sf(i nu)   (SCTK, star-weighted q average, work chain {node.pk} {node.uuid})\n")
                for x, y in zip(nu * 1000, lam): handle.write(f"{x:16.6f} {y:16.8e}\n")
            print('  wrote', table)
        if ctable and n == 0:
            muc = w @ node.outputs.kernel.get_array('mu_coulomb')
            with open(ctable, 'w') as handle:
                handle.write(f"# imag-axis table: nu[meV]  mu_C(i nu)   (SCTK, star-weighted q average, laddxc = {node.inputs.sctk.sctk.parameters.get_dict()['KEL']['laddxc']}, work chain {node.pk} {node.uuid})\n")
                for x, y in zip(nu * 1000, muc): handle.write(f"{x:16.6f} {y:16.8e}\n")
            print('  wrote', ctable)
