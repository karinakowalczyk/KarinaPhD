from firedrake import (Constant, PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, Function, as_vector, sqrt,
                       conditional, cos, pi, DistributedMeshOverlapType
                       )
from tools import Parameters, compressibleEulerEquations

'''
rising bubble over a mountain test
taken from the paper https://doi.org/10.1175/MWR-D-14-00054.1
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
L = 20000.
H = 20000.  # Height position of the model top
delx = 150
delz = 150
nlayers = H/delz+1  # horizontal layers
columns = L/delx+1  # number of columns
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = PeriodicIntervalMesh(columns, L, distribution_parameters=distribution_parameters)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=delz)

# define mountain coordinates
a = 1000.
xc = L/2.
x, z = SpatialCoordinate(mesh)
hm = 250.
zs = hm*a**2/((x-xc)**2 + a**2)
xexpr = as_vector([x, z + ((H-z)/H)*zs])
Vc = mesh.coordinates.function_space()
f_mesh = Function(Vc).interpolate(as_vector([x, z + ((H-z)/H)*zs]))
mesh.coordinates.assign(f_mesh)


# initialise background temperature
Tsurf = 300.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Constant(Tsurf)

# define bubble as temperature perturbation
xc = L/2
xr = 2000.
zc = 4500.
zr = 2000.

Lr = sqrt(((x-xc)/xr)**2 + ((z-zc)/zr)**2)
thetab_pert = conditional(Lr > 1., 0., 2*(cos(pi*Lr/2))**2)

# set up fem spaces
vertical_degree = 1
horizontal_degree = 1

Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree)

Problem.H = H  # needed for sponge_fct
# Problem.u0 = u0
Problem.solver_params = sparameters_star
Problem.path_out = "../Results/bubble_over_mountain/mountain_bubble"
Problem.thetab = thetab
Problem.theta_init_pert = thetab_pert
Problem.sponge_fct = True

dt = 4.
tmax = 1000.
dumpt = 10.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
