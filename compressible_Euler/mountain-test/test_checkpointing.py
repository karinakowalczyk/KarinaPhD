from firedrake import *

mesh = UnitSquareMesh(20,20)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
u = Function(V).interpolate(sin(x)+cos(y))
w = Function(V).interpolate(exp(x*y))

with CheckpointFile("testfile.h5", 'w') as file:
    file.save_mesh(mesh)
    file.save_function(u)
    file.save_function(w)
