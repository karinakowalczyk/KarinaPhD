from firedrake import *
import h5py
import numpy as np


data_matrix = np.random.uniform(-1, 1, size=(10, 3))
with h5py.File("test_file.h5", "w") as data_file:
    data_file.create_dataset("dataset_name", data=data_matrix)
'''
mesh = UnitSquareMesh(20,20)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
u = Function(V).interpolate(sin(x)+cos(y))
w = Function(V).interpolate(exp(x*y))

with CheckpointFile("testfile.h5", 'w') as file:
    file.save_mesh(mesh)
    file.save_function(u)
    file.save_function(w)
'''