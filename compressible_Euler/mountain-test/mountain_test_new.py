from firedrake import *
from tools import *
from petsc4py import PETSc



solver_parameters={'snes_type': 'newtonls',
                         'ksp_type': 'preonly',
                         'pc_type': 'lu'}

sparameters_exact = { 
                   'snes_monitor': None,
                   'snes_stol': 1e-50,
                   #'snes_view': None,
                   #'snes_type' : 'ksponly',
                   'ksp_monitor_true_residual': None,
                   "ksp_atol": 1e-20,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None,
                   "ksp_type" : "preonly",
                   "pc_type" : "lu",
                   #"pc_factor_mat_solver_type": "mumps"
                   }

#requires a mesh hierarchy
# multigrid solver
sparameters_mg = {
        "snes_monitor": None,
        "snes_stol": 1e-8,
        "snes_converged_reason" : None,
        "mat_type": "aij",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "python",
        'mg_levels_pc_python_type': 'firedrake.ASMStarPC',
        "mg_levels_pc_star_sub_pc_type": "lu",
        'mg_levels_pc_star_construct_dim': '0',
        'mg_levels_pc_star_sub_pc_factor_mat_solver_type' : 'mumps',
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    }

sparameters_vanka = {
        "snes_monitor": None,
        "snes_stol": 1e-8,
        "snes_converged_reason" : None,
        "mat_type": "aij",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        #"mg_levels_ksp_type": "gmres",
        #"mg_levels_ksp_max_it": 3,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "python",
        "assembled_pc_python_type": "firedrake.ASMVankaPC",
        "assembled_pc_vanka_construct_dim": 0,
        "assembled_pc_vanka_sub_pc_type": "lu",   # but this is actually the default.
        "assembled_pc_vanka_sub_pc_factor_mat_solver_type" : 'mumps',
    }

sparameters_star = {
        "snes_monitor": None,
        "snes_stol": 1e-50,
        "snes_converged_reason" : None,
        "mat_type": "aij",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        #"mg_levels_ksp_type": "gmres",
        #"mg_levels_ksp_max_it": 3,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "python",
        "assembled_pc_python_type": "firedrake.ASMStarPC",
        "assembled_pc_star_construct_dim": 0,
        "assembled_pc_star_sub_pc_type": "lu",   # but this is actually the default.
        "assembled_pc_star_sub_pc_factor_mat_solver_type" : 'mumps',
    }

sparameters_test = {"snes_monitor": None,
        "snes_stol": 1e-50,
        'snes_type':"newtontrdc",
        "snes_converged_reason" : None,
        "mat_type": "aij",
        "ksp_type" : "gmres",
        "pc_type" : "ilu",
        #"pc_factor_mat_solver_type": "mumps",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
}


sparameters = {
    "snes_converged_reason": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_factor_mat_solver_type":"mumps"
}


dT = Constant(1)  # to be set later
parameters = Parameters()
g = parameters.g
c_p = parameters.cp


# set mesh parameters
L = 100000.
H = 30000.  # Height position of the model top
delx = 500*2
delz = 300*2
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns

# build mesh
m = PeriodicIntervalMesh(columns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

# set terrain-following coordinates

coord = SpatialCoordinate(ext_mesh)
a = 1000.
xc = L/2.
x, z = SpatialCoordinate(ext_mesh)
hm = 250.
zs = hm*a**2/((x-xc)**2 + a**2)
xexpr = as_vector([x, z + ((H-z)/H)*zs])

#Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
#new_coords = Function(Vc).interpolate(xexpr)
#mesh = Mesh(new_coords)

mesh = ext_mesh
Vc = mesh.coordinates.function_space()
f_mesh = Function(Vc).interpolate(as_vector([x,z + ((H-z)/H)*zs]) )
mesh.coordinates.assign(f_mesh)



# set up fem spaces
vertical_degree=1
horizontal_degree=1

# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 288.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)


u0 = as_vector([10., 0.])
Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree)

Problem.H = H # edit later in class
Problem.u0 = u0
Problem.dT = Constant(5.)
Problem.solver_params = sparameters_mg
Problem.path_out = "../../Results/compEuler/mountain-test-adv/mountain-adv"
Problem.thetab = thetab
Problem.theta_init_pert = 0
Problem.sponge_fct = True

#Problem.initilise_rho_lambdar_hydr_balance

dt = 5.
tmax = 15000.
dumpt = 10.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
