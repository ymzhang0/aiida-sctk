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
