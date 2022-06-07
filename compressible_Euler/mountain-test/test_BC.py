from firedrake import *
from tools import *


u0 = as_vector([10.0, 0])

L = 100000.
H = 30000.  # Height position of the model top
delx = 500
delz = 300
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns
m = PeriodicIntervalMesh(columns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
coord = SpatialCoordinate(ext_mesh)
x = Function(Vc).interpolate(as_vector([coord[0], coord[1]]))
a = 5000.
xc = L/2.
x, z = SpatialCoordinate(ext_mesh)
hm = 250.
lamb = 4000.
zs = hm*exp(-((x-xc)/a)**2) * (cos(pi*(x-xc)/lamb))**2
xexpr = as_vector([x, z + ((H-z)/H)*zs])
new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)

V, _, _, _, T = build_spaces(mesh, 1, 1)
U_bc = apply_BC_def_mesh(u0, V, T)
u_bc, lambdar_bc = U_bc.split()

file = File("u0BC.pvd")
file.write(u_bc)

