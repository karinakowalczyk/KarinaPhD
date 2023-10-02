from firedrake import *
import sympy as sp
from tools.physics import * 
from tools.compressible_Euler import *
import numpy as np

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

params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

x, z = sp.symbols('x z')
t =0.

L = 1.
nx = 2*64

m = PeriodicIntervalMesh(64, L)
mesh = ExtrudedMesh(m, nx, layer_height=L/nx, periodic=True)
x, y = SpatialCoordinate(mesh)

uc = 1.
vc = 1.

xc = 0.5 + t*uc
yc = 0.5 + t*vc

R = 0.5*0.4 #radius of vortex
r = sqrt((x - xc)**2 + (y - yc)**2)/R
phi = atan((y - yc)/(x-xc))

rhoc = Constant(1.)
rhob = conditional(r>=1, 1.,  1- 0.5*(1-r**2)**6)
ux= conditional(r>=1, 1.,  -1024*sin(phi)*(1-r**6)*r**6 + 1.)
uy = conditional(r>=1, 1., 1024*sin(phi)*(1-r**6)*r**6 + 1.)
u0 = as_vector([ux, uy])

'''
coe = np.zeros((25))
coe[0]  =     1.0 / 24.0
coe[1]  = -   6.0 / 13.0
coe[2]  =    18.0 /  7.0
coe[3]  = - 146.0 / 15.0
coe[4]  =   219.0 / 8.0
coe[5]  = - 966.0 / 17.0
coe[6]  =   731.0 /  9.0
coe[7]  = -1242.0 / 19.0
coe[8]  = -  81.0 / 40.0
coe[9]  =   64.
coe[10] = - 477.0 / 11.0
coe[11] = -1032.0 / 23.0
coe[12] =   737.0 / 8.0
coe[13] = - 204.0 /  5.0
coe[14] = - 510.0 / 13.0
coe[15] =  1564.0 / 27.0
coe[16] = - 153.0 /  8.0
coe[17] = - 450.0 / 29.0
coe[18] =   269.0 / 15.0
coe[19] = - 174.0 / 31.0
coe[20] = -  57.0 / 32.0
coe[21] =    74.0 / 33.0
coe[22] = -  15.0 / 17.0
coe[23] =     6.0 / 35.0
coe[24] =  -  1.0 / 72.0
'''

coe = np.zeros((25))
coe[0]  =     1.0 / 24.0
coe[1]  = -   6.0 / 13.0
coe[2]  =    15.0 /  7.0
coe[3]  = -  74.0 / 15.0
coe[4]  =    57.0 / 16.0
coe[5]  =   174.0 / 17.0
coe[6]  = - 269.0 /  9.0
coe[7]  =   450.0 / 19.0
coe[8]  =  1071.0 / 40.0
coe[9]  = -1564.0 / 21.0
coe[10] =   510.0 / 11.0
coe[11] =  1020.0 / 23.0
coe[12] = -1105.0 / 12.0
coe[13] =   204.0 /  5.0
coe[14] =   510.0 / 13.0
coe[15] = -1564.0 / 27.0
coe[16] =   153.0 /  8.0
coe[17] =   450.0 / 29.0
coe[18] = - 269.0 / 15.0
coe[19] =   174.0 / 31.0
coe[20] =    57.0 / 32.0
coe[21] = -  74.0 / 33.0
coe[22] =    15.0 / 17.0
coe[23] = -   6.0 / 35.0
coe[24] =     1.0 / 72.0


const_coe = np.zeros((13))
const_coe[0] =    1.0 / 24
const_coe[1] = -  6.0 / 13
const_coe[2] =   33.0 / 14
const_coe[3] = - 22.0 / 3
const_coe[4] =  495.0 / 32
const_coe[5] = -396.0 / 17
const_coe[6] = + 77.0 / 3
const_coe[7] = -396.0 / 19
const_coe[8] =   99.0 / 8
const_coe[9] = -110.0 / 21
const_coe[10] = + 3.0 / 2
const_coe[11] = - 6.0 / 23
const_coe[12] = + 1.0 / 48

'''
p_inner =1.0/24.0 * r**12

for k in range(1,25):
    p_inner += coe[k] * (r**(k+12)-1)
'''
p = 0
for ip in range(25):
    p += coe[ip] * (r**(12+ip) - 1.0)
for ip in range(13):
    p += const_coe[ip] * (r**(12+ip) - 1.0)
    
R0 = 287.
gamma = 1.4
pref = Parameters.p_0

p = 1 + conditional(r>=0, 1, p)

#thetab = thermodynamics_theta(p, rhob)

#thetab = pref/(R0*rhob)*(pi**(1/(gamma -1)))

T = p/(rhob*R0)
thetab = T*(pref/p)**0.286



Problem = compressibleEulerEquations(mesh, vertical_degree=1, horizontal_degree=1)
Problem.g = 0.
Problem.H = L # edit later in class
Problem.u0 = u0
Problem.rhob = rhob
Problem.solver_params = sparameters_star
Problem.path_out = "../Results/convergence"
Problem.thetab = thetab
Problem.theta_init_pert = 0
Problem.sponge_fct = False
Problem.checkpointing = False
#Problem.checkpoint_path = "checkpointNHMW.h5"

dt = 0.45/nx
tmax = 100.
dumpt = 2.

Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt, hydrostatic_balance_background=False)
