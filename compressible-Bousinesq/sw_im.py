import firedrake as fd

#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 24.')
parser.add_argument('--dumpt', type=float, default=1, help='Dump time in hours. Default 1.')
parser.add_argument('--gamma', type=float, default=1.0e5, help='Augmented Lagrangian scaling parameter. Default 10000.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5')
parser.add_argument('--kspinner', type=int, default=3, help='Number of ksp inner iterations')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspschur', type=int, default=3, help='Number of KSP iterations on the Schur complement.')
parser.add_argument('--kspmg', type=int, default=3, help='Number of KSP iterations in the MG levels.')
parser.add_argument('--tlblock', type=str, default='mg', help='Solver for the velocity-velocity block. mg==Multigrid with patchPC, lu==direct solver with MUMPS, patch==just do a patch smoother.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
args = parser.parse_known_args()
args = args[0]

if args.show_args:
    print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
#distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.FACET, 2)}
if args.tlblock == "mg":
    basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                        refinement_level=base_level, degree=deg,
                                        distribution_parameters = distribution_parameters)
    mh = fd.MeshHierarchy(basemesh, nrefs)
    for mesh in mh:
        x = fd.SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    mesh = mh[-1]
else:
    mesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=args.ref_level, degree=deg,
                                    distribution_parameters = distribution_parameters)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDFM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)
gamma0 = args.gamma
gamma = fd.Constant(gamma0)

# Set up the exponential operator
operator_in = fd.Function(W)
u_in, eta_in = fd.split(operator_in)

# D = eta + b

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Unp1)
uh = 0.5*(u0 + u1)
hh = 0.5*(h0 + h1)
n = fd.FacetNormal(mesh)
Upwind = 0.5 * (fd.sign(fd.dot(uh, n)) + 1)


def both(u):
    return 2*fd.avg(u)


K = 0.5*fd.inner(uh, uh)
uup = 0.5 * (fd.dot(uh, n) + abs(fd.dot(uh, n)))
dT = fd.Constant(0.)
dS = fd.dS

vector_invariant = True
if vector_invariant:
    eqn = (
        fd.inner(v, u1 - u0)*dx + dT*fd.inner(v, f*perp(uh))*dx
        - dT*fd.inner(perp(fd.grad(fd.inner(v, perp(uh)))), uh)*dx
        + dT*fd.inner(both(perp(n)*fd.inner(v, perp(uh))),
                      both(Upwind*uh))*dS
        - dT*fd.div(v)*(g*(hh + b) + K)*dx
        + phi*(h1 - h0)*dx
        - dT*fd.inner(fd.grad(phi), uh)*hh*dx
        + dT*fd.jump(phi)*(uup('+')*hh('+')
                           - uup('-')*hh('-'))*dS
        # the extra bit
        + gamma*(fd.div(v)*(h1 - h0)*dx
                 - dT*fd.inner(fd.grad(fd.div(v)), uh)*hh*dx
                 + dT*fd.jump(fd.div(v))*(uup('+')*hh('+')
                                          - uup('-')*hh('-'))*dS)
        )

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": args.kspschur,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

sparameters["fieldsplit_1"] = bottomright

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

topleft_MG = {
    "ksp_type": "preonly",
    "ksp_max_it": 3,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": args.kspmg,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}

topleft_smoother = {
    "ksp_type": "gmres",
    "ksp_max_it": 50,
    "ksp_monitor": None,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": False,
    "patch_pc_patch_sub_mat_type": "seqaij",
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_multiplicative": False,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_pc_patch_construct_dim": 0,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
}

if args.tlblock == "mg":
    sparameters["fieldsplit_0"] = topleft_MG
elif args.tlblock == "patch":
    sparameters["fieldsplit_0"] = topleft_smoother
else:
    assert(args.tlblock=="lu")
    sparameters["fieldsplit_0"] = topleft_LU

dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
ctx = {"mu": g*dt/gamma/2}
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters,
                                        appctx=ctx)
dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

x = fd.SpatialCoordinate(mesh)
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

u0, h0 = Un.split()
u0.assign(un)
h0.assign(etan + H - b)

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob, name="qsolver",
                                     solver_parameters=qparams)

name = "sw_imp"
file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)
Unp1.assign(Un)

print('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
