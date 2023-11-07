from firedrake import *

mesh = UnitSquareMesh(2, 2, name="meshA")
V = FunctionSpace(mesh, "CG", 1)
f = Function(V, name="f")
x, y = SpatialCoordinate(mesh)

with CheckpointFile("example_timestepping.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    for i in range(4):
        f.interpolate(x * i)
        afile.save_function(f, idx=i)
with CheckpointFile("example_timestepping.h5", 'r') as afile:
    mesh = afile.load_mesh("meshA")
    for i in range(4):
        f = afile.load_function(mesh, "f", idx=i)
        print("norm for function ", i, "is ", assemble(f*dx))

with CheckpointFile("../mountain-test/checkpointNHMW.h5", 'r') as file: 
    mesh = file.load_mesh("meshA")
'''
# set mesh parameters
L = 144000.
H = 35000.  # Height position of the model top
delx = 400*2
delz = 250*2
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns

# build mesh
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)


#build spaces
horizontal_degree = 1
vertical_degree = 1

# horizontal base spaces
cell = mesh._base_mesh.ufl_cell().cellname()
S1 = FiniteElement("CG", cell, horizontal_degree + 1)  # EDIT: family replaced by CG (was called with RT before)
S2 = FiniteElement("DG", cell, horizontal_degree, variant="equispaced")

# vertical base spaces
T0 = FiniteElement("CG", interval, vertical_degree + 1, variant="equispaced")
T1 = FiniteElement("DG", interval, vertical_degree, variant="equispaced")

# trace base space
Tlinear = FiniteElement("CG", interval, 1)

# build spaces V2, V3, Vt
V2h_elt = HDiv(TensorProductElement(S1, T1))
V2t_elt = TensorProductElement(S2, T0)
V3_elt = TensorProductElement(S2, T1)
V2v_elt = HDiv(V2t_elt)

V2v_elt_Broken = BrokenElement(HDiv(V2t_elt))
V2_elt = EnrichedElement(V2h_elt, V2v_elt_Broken)
VT_elt = TensorProductElement(S2, Tlinear)

remapped = WithMapping(V2_elt, "identity")

V0 = FunctionSpace(mesh, remapped, name="new_velocity")
V1 = FunctionSpace(mesh, V3_elt, name="DG")  # pressure space
V2 = FunctionSpace(mesh, V2t_elt, name="Temp")
u0 = as_vector([10., 0.])

u = Function(V0).project(u0)
rho = Function(V1).interpolate(Constant(1.))

with CheckpointFile("exampleNEW.h5", 'w') as afile:
    afile.save_function(u)
    afile.save_function(rho)
'''