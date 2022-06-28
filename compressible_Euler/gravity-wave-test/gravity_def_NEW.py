from firedrake import *
from tools import *
import sys

'''
    version of "mountain_non-hydrostatic in gusto for new velocity space
    without making use of gusto
'''


# set up mesh and parameters for main computations
dT = Constant(0)  # to be set later
parameters = Parameters()
g = parameters.g
c_p = parameters.cp


# build volume mesh
L = 3.0e5
H = 1.0e4  # Height position of the model top
delx = 1000
delz = 1000
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
new_coords = Function(Vc).interpolate(as_vector([x,y -(4/H**2) *(sin(2*pi*x/L))*((y-H/2)**2 - H**2/4)]  ))
mesh = Mesh(new_coords)
H = Constant(H)

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
    #"assembled_pc_star_sub_pc_factor_mat_ordering_type": "rcm",
    #"assembled_pc_star_sub_pc_factor_nonzeros_along_diagonal": 1e-8,
}

name = "../../Results/compEuler/gravity-wave/flat/gravity_def"

H = Constant(H)



# initialise background temperature 
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)



# initialise functions for full Euler solver
xc = L/2
a = Constant(5000.)
thetab_pert = 0.01*sin(pi*z/H)/(1+(x-xc)**2/a**2)


u0 = as_vector([20.0, 0.0])


vertical_degree = 1
horizontal_degree = 1
Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree)

Problem.H = H # edit later in class
Problem.u0 = u0
Problem.dT = Constant(12.)
Problem.solver_params = sparameters_star
Problem.path_out = "../../Results/compEuler/gravity-wave/def/gravity_def"
Problem.thetab = thetab
Problem.theta_init_pert = thetab_pert
#Problem.sponge_fct = True

#Problem.initilise_rho_lambdar_hydr_balance

dt = 12.
tmax = 6000.
dumpt = 24.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)