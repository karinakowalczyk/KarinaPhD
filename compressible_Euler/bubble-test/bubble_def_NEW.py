from firedrake import *
from tools import *

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
L = 10000.
H = 6400.  # Height position of the model top
delx = 50.
delz = 50.
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
new_coords = Function(Vc).interpolate(as_vector([x,y -(4/H**2) *(sin(2*pi*x/L))*((y-H/2)**2 - H**2/4)]  ))
mesh = Mesh(new_coords)

H = Constant(H)

vertical_degree = 1
horizontal_degree = 1

# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Constant(Tsurf)

# initialise functions for full Euler solver
xc = L/2
xr = 2000.
zc = 2000.
zr = 2000.

Lr = sqrt(((x-xc)/xr)**2 + ((z-zc)/zr)**2)

delT = conditional(Lr > 1., 0., 2*(cos(pi*Lr/2))**2)
thetab_pert = delT


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
 
 

Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree)

Problem.H = H # edit later in class
#Problem.u0 = u0
Problem.dT = Constant(2.)
Problem.solver_params = sparameters_star
Problem.path_out = "../../Results/compEuler/bubble-test/bubble"
Problem.thetab = thetab
Problem.theta_init_pert = thetab_pert
Problem.sponge_fct = True

#Problem.initilise_rho_lambdar_hydr_balance
dt = 1.
dumpt = 4.
tdump = 0.
dT.assign(dt)
tmax = 2000.


Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)

