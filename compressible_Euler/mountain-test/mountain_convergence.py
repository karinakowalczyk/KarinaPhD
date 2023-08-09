import firedrake as fd
from tools.physics import Parameters
from tools.compressible_Euler import compressibleEulerEquations
import sympy as sp

'''
testing convergence by decreasing delx
flow over mountain test case taken from  
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
    'assembled_pc_star_sub_pc_factor_mat_solver_type': 'mumps',
    # "assembled_pc_star_sub_pc_factor_mat_ordering_type": "rcm",
    # "assembled_pc_star_sub_pc_factor_nonzeros_along_diagonal": 1e-8,
}
dT = fd.Constant(1)  # to be set later
parameters = Parameters()
g = parameters.g
c_p = parameters.cp

k = 1
k_str = str(k)

H = 35e3  # Height position of the model top
L = 144e3

#MANUFACTURED SOLUTION in sympy
x, z = sp.symbols('x z')

Ax = 1.5
Az = 0.6
Bx = 1.
Bz = 0.9

sin = sp.sin
cos = sp.cos
pi = sp.pi
exp = sp.exp

ux = exp(cos(Ax*pi*x/L)+sin(Az*pi*z/H))
uz = exp(sin(Bx*pi*x/L)+cos(Bz*pi*z/H))

ugradu_x = ux*ux.diff(x) + uz*ux.diff(z)
ugradu_z = ux*uz.diff(x) + uz*uz.diff(z)

rho = exp(sin(pi*x/L)+cos(1.5*pi*z/H))
theta = exp(cos(pi*L)+sin(2*pi*z/H))

R_d = 287.  # Gas constant for dry air (J/kg/K)
kappa = 2.0/7.0  # R_d/c_p
p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)

Pi = (R_d*rho*theta/p_0)**(kappa/(1-kappa))
theta_Pi_diffx = theta*Pi.diff(x)
theta_Pi_diffz = theta*Pi.diff(z)
rhs_u_x = ugradu_x + theta_Pi_diffx
rhs_u_z = ugradu_z + theta_Pi_diffz

rhs_rho = (rho*ux).diff(x) + (rho*uz).diff(z)

rhs_theta = ux*theta.diff(x) + uz*theta.diff(z)




#delx = 400*2
#delz = 250*2
nlayers = 6*(2**k)  # horizontal layers
ncolumns = 16*(2**k) # number of columns

distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
m = fd.PeriodicIntervalMesh(ncolumns, L, distribution_parameters =
                            distribution_parameters)
m.coordinates.dat.data[:] -= L/2

# build mesh
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, name="mesh")

# making a mountain out of a molehill
a = 1000.
xc = 0.
x, z = fd.SpatialCoordinate(mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)
xexpr = fd.as_vector([x, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)

#comvert manufactured solution to ufl
cos = fd.cos
sin = fd.sin
pi = fd.pi
exp = fd.exp
x, z = fd.SpatialCoordinate(mesh)
rhs_rho = eval(str(rhs_rho))
rhs_theta = eval(str(rhs_theta))
rhs_u_x = eval(str(rhs_u_x))
rhs_u_z = eval(str(rhs_u_z))
rhs_u = fd.as_vector([rhs_u_x, rhs_u_z])


# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
N = parameters.N
x, z = fd.SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)


u0 = fd.as_vector([10., 0.])

# set up fem spaces
vertical_degree = 1
horizontal_degree = 1

Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree, rhs = True)
Problem.rhs_u = rhs_u
Problem.rhs_rho = rhs_rho
Problem.rhs_theta = rhs_theta

Problem.H = H  # edit later in class
Problem.u0 = u0
Problem.solver_params = sparameters_star
Problem.path_out = "../Results/convergence/" + k_str + "/mountainNH"
Problem.thetab = thetab
Problem.theta_init_pert = 0
Problem.sponge_fct = True
Problem.checkpointing = True
Problem.checkpoint_path = "checkpointNHMW" + k_str + ".h5"

dt = 5.
tmax = 3600.
dumpt = 10*5.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
