from firedrake import *
from tools import build_spaces

def apply_BC_def_mesh(u0, V, T):

    """
    apply boundary condition u dot n = 0 to the initial velocity
    in the velocity FEM space V, where V is  the new broken space and
    T the trace space,
    project by solving the according PDE
    """

    W = V*T
    w, mu = TestFunctions(W)
    U = Function(W)
    u, lambdar = split(U)
    n = FacetNormal(V.mesh())

    a = (inner(w,u)*dx - inner(w,u0)*dx 
         + jump(w,n)*lambdar('+')*dS_h
         + inner(w, n)*lambdar*ds_tb
         + jump(u,n)*mu('+')*dS_h
         + inner(u, n)*mu*ds_tb
        )
    L = 0
    sparameters_exact = {"mat_type": "aij",
                   'snes_monitor': None,
                   #'snes_stol': 1e-50,
                   #'snes_view': None,
                   #'snes_type' : 'ksponly',
                   'ksp_monitor_true_residual': None,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None,
                   "ksp_type" : "preonly",
                   "pc_type" : "lu",
                   "pc_factor_mat_solver_type": "mumps"}

    problem = NonlinearVariationalProblem(a, U)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sparameters_exact)

    solver.solve()

    return U


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

