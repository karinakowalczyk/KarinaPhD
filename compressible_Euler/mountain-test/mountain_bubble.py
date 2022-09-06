from firedrake import *
from tools import *
import numpy as np


sparameters_exact = { "mat_type": "aij",
                   'snes_monitor': None,
                   'snes_stol': 1e-50,
                   #'snes_view': None,
                   #'snes_type' : 'ksponly',
                   'ksp_monitor_true_residual': None,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None,
                   "ksp_type" : "preonly",
                   "pc_type" : "lu",
                   "pc_factor_mat_solver_type": "mumps"
                   }

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

sparameters_star_2 = {
    "snes_monitor": None,
    "snes_stol": 1e-20,
    "ksp_monitor_true_residual": None,
    #"ksp_view": None,
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
    "assembled_pc_star_sub_sub_pc_type": "lu",
    #"assembled_pc_star_sub_sub_ksp_view": None,
    'assembled_pc_star_sub_sub_pc_factor_mat_solver_type' : 'mumps',
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    #"assembled_pc_star_sub_sub_pc_factor_shift_type": "NONZERO",
    "assembled_pc_star_sub_sub_pc_factor_nonzeros_along_diagonal": 1e-8,
}

sparameters_mg = {
        "snes_monitor": None,
        "snes_stol": 1e-8,
        "snes_converged_reason" : None,
        "mat_type": "aij",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_view" : None,
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
        #"mg_levels_pc_star_sub_pc_factor_mat_ordering_type": "rcm"
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    }

parameters = Parameters()
g = parameters.g
c_p = parameters.cp

# build volume mesh
L = 20000.
H = 20000.  # Height position of the model top
delx = 100
delz = 100
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
m = IntervalMesh(columns, L, distribution_parameters=distribution_parameters)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=delz)


a = 1000.
xc = L/2.
x, z = SpatialCoordinate(mesh)
hm = 1000.
zs = hm*a**2/((x-xc)**2 + a**2)
xexpr = as_vector([x, z + ((H-z)/H)*zs])
Vc = mesh.coordinates.function_space()
f_mesh = Function(Vc).interpolate(as_vector([x,z + ((H-z)/H)*zs]) )
mesh.coordinates.assign(f_mesh)

Vc = VectorFunctionSpace(mesh, "DG", 2)
xexpr = as_vector([x, z + ((H-z)/H)*zs])
new_coords = Function(Vc).interpolate(xexpr)
#mesh = Mesh(new_coords)
#mesh = ext_mesh

# set up fem spaces
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
zc = 4500.
zr = 2000.

Lr = sqrt(((x-xc)/xr)**2 + ((z-zc)/zr)**2)

delT = conditional(Lr > 1., 0., 2*(cos(pi*Lr/2))**2)
thetab_pert = delT

u0 = as_vector([10., 0.])
Problem = compressibleEulerEquations(mesh, vertical_degree, horizontal_degree, mesh_periodic = False)

Problem.H = H # edit later in class
#Problem.u0 = u0
Problem.dT = Constant(2.)
Problem.solver_params = sparameters_star
Problem.path_out = "../../Results/compEuler/mountain_bubble/mountain_bubble"
Problem.thetab = thetab
Problem.theta_init_pert = thetab_pert
Problem.sponge_fct = True

dt = 2.
tmax = 1000.
dumpt = 2.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt)
