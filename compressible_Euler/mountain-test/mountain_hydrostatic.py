from firedrake import *
from tools import *
from petsc4py import PETSc

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

dT = Constant(1)  # to be set later
parameters = Parameters()
g = parameters.g
c_p = parameters.cp


nlayers = 100  # horizontal layers 50./100 = 0.5 = 2*250
base_columns = 60  # number of columns 240000/60 = 4000 = 2*2000
L = 240e3
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

m = PeriodicRectangleMesh(base_columns, ny=1, Lx=L, Ly=1.0e-3*L,
                             direction="both",
                             quadrilateral=True,
                             distribution_parameters=distribution_parameters)
m.coordinates.dat.data[:,0] -= L/2
m.coordinates.dat.data[:,1] -= 1.0e-3*L/2


# build volume mesh
H = 50e3  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = FacetNormal(mesh)

# making a mountain out of a molehill
a = 10000.
xc = 0.
x, y, z = SpatialCoordinate(mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

xexpr = as_vector([x, y, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)

#Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
#new_coords = Function(Vc).interpolate(xexpr)
#mesh = Mesh(new_coords)

Vc = mesh.coordinates.function_space()
f_mesh = Function(Vc).interpolate(as_vector([x, y, z + ((H-z)/H)*zs]))
mesh.coordinates.assign(f_mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
# Isothermal T = theta*pi is constant
# pi = T/theta => pi_z = -T/theta^2 theta_z
# so -g = cp theta pi_z = -T/theta theta_z
# so theta_z = theta*g/T/cp
# i.e. theta = theta_0 exp(g*z/T/cp)
Tsurf = Constant(250.)
N = g/sqrt(c_p*Tsurf)
thetab = Tsurf*exp(N**2*z/g)


# set up fem spaces
vertical_degree=1
horizontal_degree=1

# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 288.
N = parameters.N
thetab = Tsurf*exp(N**2*z/g)

u0 = as_vector([Constant(20.0),
                Constant(0.0),
                Constant(0.0)])

Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree, slice_3D=True)
Problem.H = H # edit later in class
Problem.u0 = u0
Problem.dT = Constant(20.)
Problem.solver_params = sparameters_star
Problem.path_out = "../../Results/compEuler/mountain-hydrostatic/mountain_hydrostatic"
Problem.thetab = thetab
Problem.theta_init_pert = 0
Problem.sponge_fct = True

#Problem.initilise_rho_lambdar_hydr_balance

dt = 20.
tmax = 15000.
dumpt = 20.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
