"""Provenance-tracked replacement for SCTK twingrid.x."""
from itertools import product
from aiida import orm
from aiida.engine import calcfunction


def twin_points(mesh, shift):
    """Keep the shell script's order, weights and intentional duplicate points."""
    if len(mesh) != 3 or any(type(n) is not int or n < 1 for n in mesh):
        raise ValueError('mesh must contain three positive integers')
    if len(shift) != 3 or any(type(s) is not int or s not in (0, 1) for s in shift):
        raise ValueError('shift must contain three integers equal to zero or one')
    points = []
    for offset in ((0, 0, 0), shift):
        for indices in product(*(range(n) for n in mesh)):
            points.append([(2*i+s)/(2*n) - int(2*i+s >= n)
                           for i, s, n in zip(indices, offset, mesh)])
    return points


@calcfunction
def generate_twin_kpoints(mesh):
    """Expand a mesh and its half-grid offset into SCTK's doubled point list."""
    dimensions, offset = mesh.get_kpoints_mesh()
    shift = [round(value * 2) for value in offset]
    if any(abs(value * 2 - integer) > 1.e-12 or integer not in (0, 1)
           for value, integer in zip(offset, shift)):
        raise ValueError('mesh offset must contain only 0 or 0.5')
    points = twin_points(dimensions, shift)
    result = orm.KpointsData()
    result.set_kpoints(points, weights=[1.0]*len(points))
    return result


def count_irreducible_qpoints(structure, mesh, shift=(0, 0, 0), symprec=1.e-5):
    """Number of irreducible q points of a uniform grid (crystal symmetry and time reversal).

    This is the number of q points ``sctk.x`` loops over for ``KEL.nq1..3`` and ``CONTROL.lshift_q``;
    it is only used to split the q points over several jobs, and the work chain checks it against
    the number that ``sctk.x`` itself reports.
    """
    import numpy as np
    import spglib

    atoms = structure.get_ase()
    cell = (atoms.get_cell().array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    mapping, _ = spglib.get_ir_reciprocal_mesh(
        list(mesh), cell, is_shift=[int(bool(s)) for s in shift], is_time_reversal=True, symprec=symprec)
    return int(len(np.unique(mapping)))


def split_qpoints(first, last, per_job):
    """Split the inclusive q-point index range ``first..last`` into chunks of at most ``per_job``."""
    if per_job < 1 or last < first:
        raise ValueError('need per_job >= 1 and last >= first')
    return [[start, min(start + per_job - 1, last)] for start in range(first, last + 1, per_job)]


@calcfunction
def merge_kel_kernels(**kernels):
    """Merge the ``kernel`` arrays of kel jobs that each computed a subset of the q points."""
    import numpy as np

    nodes = list(kernels.values())
    frequencies = nodes[0].get_array('frequencies')
    indices = np.concatenate([node.get_array('qpoint_indices') for node in nodes])
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('the same q point was computed by more than one kel job')
    order = np.argsort(indices)
    merged = orm.ArrayData()
    merged.set_array('frequencies', frequencies)
    for name in nodes[0].get_arraynames():
        if name in ('frequencies', 'qpoints_cartesian'):
            continue
        if not all(name in node.get_arraynames() for node in nodes):
            continue
        for node in nodes:
            if not np.allclose(node.get_array('frequencies'), frequencies):
                raise ValueError('kel jobs used different frequency grids')
        merged.set_array(name, np.concatenate([node.get_array(name) for node in nodes])[order])
    if 'qpoints_cartesian' in nodes[0].get_arraynames():
        merged.set_array('qpoints_cartesian', nodes[0].get_array('qpoints_cartesian'))
    merged.base.attributes.set_many(dict(nodes[0].base.attributes.all))
    return merged
