from firedrake import *
from tools.physics import Parameters
from tools.compressible_Euler import compressibleEulerEquations

'''
flow over mountain test case taken from  
    https://doi.org/10.1002/qj.603
in the non-hydrostatic regime
'''


sparameters_star = {
    "snes_monitor": None,
    "snes_stol": 1e-100,
    "snes_rtol": 1e-100,
    "snes_atol": 1e-8,
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
    'assembled_pc_star_sub_pc_factor_mat_solver_type': 'mumps',
    # "assembled_pc_star_sub_pc_factor_mat_ordering_type": "rcm",
    # "assembled_pc_star_sub_pc_factor_nonzeros_along_diagonal": 1e-8,
}

dT = Constant(1)  # to be set later
parameters = Parameters()
g = parameters.g
c_p = parameters.cp


H = 35e3  # Height position of the model top
L = 144e3
delx = 400*2
delz = 250*2
nlayers = H/delz  # horizontal layers
ncolumns = L/delx  # number of columns

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = PeriodicIntervalMesh(ncolumns, L, distribution_parameters =
                            distribution_parameters)
m.coordinates.dat.data[:] -= L/2

# build mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, name="mesh")

# making a mountain out of a molehill
a = 1000.
xc = 0.
x, z = SpatialCoordinate(mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)
xexpr = as_vector([x, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)

# set terrain-following coordinates
#coord = SpatialCoordinate(ext_mesh)
#a = 1000.
#xc = L/2.
#x, z = SpatialCoordinate(ext_mesh)
#hm = 1.
#zs = hm*a**2/((x-xc)**2 + a**2)
#xexpr = as_vector([x, z + ((H-z)/H)*zs])

#mesh = ext_mesh
#Vc = mesh.coordinates.function_space()
#f_mesh = Function(Vc).interpolate(xexpr)
#mesh.coordinates.assign(f_mesh)


# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)


u0 = as_vector([10., 0.])

# set up fem spaces
vertical_degree = 1
horizontal_degree = 1

Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree)

Problem.H = H  # edit later in class
Problem.u0 = u0
Problem.solver_params = sparameters_star
Problem.path_out = "../Results/nonhydrostatic_mountain/mountainNH"
Problem.thetab = thetab
Problem.theta_init_pert = 0
Problem.sponge_fct = True
Problem.checkpointing = True
Problem.checkpoint_path = "checkpointNHMW.h5"

dt = 5.
tmax = 9000.
dumpt = 10*5.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
