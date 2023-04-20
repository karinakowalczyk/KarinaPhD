from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, Function, as_vector, exp,
                       DistributedMeshOverlapType, Constant,
                       cos, pi
                       )
from tools.physics import Parameters
from tools.compressible_Euler import compressibleEulerEquations

'''
flow over a Schaer mountain profile 
test case taken from  
    https://doi.org/10.1002/qj.603
in the non-hydrostatic regime
'''


sparameters_star = {
    "snes_monitor": None,
    "snes_stol": 1e-20,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "snes_converged_reason": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_pc_type": "lu",
    'assembled_pc_star_sub_pc_factor_mat_solver_type' : 'mumps',
    # "assembled_pc_star_sub_pc_factor_mat_ordering_type": "rcm",
    # "assembled_pc_star_sub_pc_factor_nonzeros_along_diagonal": 1e-8,
}

parameters = Parameters()
g = parameters.g
c_p = parameters.cp

# build volume mesh
L = 100000.
H = 30000.  # Height position of the model top
delx = 500
delz = 300
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = PeriodicIntervalMesh(columns, L, distribution_parameters=distribution_parameters)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=delz)

a = 5000.
xc = L/2.
x, z = SpatialCoordinate(mesh)

hm = 250.
hm = Constant(hm)
lamb = 4000.
zs = hm*exp(-((x-xc)/a)**2) * (cos(pi*(x-xc)/lamb))**2

Vc = mesh.coordinates.function_space()
f_mesh = Function(Vc).interpolate(as_vector([x, z + ((H-z)/H)*zs]))
mesh.coordinates.assign(f_mesh)

# set up fem spaces
vertical_degree = 1
horizontal_degree = 1

# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 288.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)


u0 = as_vector([10., 0.])
Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree)

Problem.H = H  # edit later in class
Problem.u0 = u0
Problem.solver_params = sparameters_star
Problem.path_out = "../Results/mountain_Schar"
Problem.thetab = thetab
Problem.theta_init_pert = 0
Problem.sponge_fct = True

dt = 8.
tmax = 8000.
dumpt = 8.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
