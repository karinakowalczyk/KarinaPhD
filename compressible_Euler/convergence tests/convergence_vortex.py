from firedrake import *
import sympy as sp
from tools.physics import * 
from tools.compressible_Euler import *
import numpy as np

sparameters_star = {
    "snes_monitor": None,
    "snes_atol": 1e-50,
    "snes_rtol": 1e-8,
    "snes_stol": 1e-50,
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
    "assembled_pc_star_sub_sub_pc_type": "ilu",
    #'assembled_pc_star_sub_sub_pc_factor_mat_solver_type': 'mumps',
    #"assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    # "assembled_pc_star_sub_pc_factor_nonzeros_along_diagonal": 1e-8,
}

sparameters = {
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
        'assembled_pc_star_sub_sub_pc_factor_mat_solver_type': 'mumps',
        "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm"
    }

params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

def exact_solution (x, y, t):

    xc = L/2 + t*uc
    yc = L/2 + t*vc

    #if xc >+ L:
    #    xc -=L
    #if yc>=L:
    #    yc -=L

    R = 0.4*L #radius of vortex
    distx = conditional(abs(x-xc)>L/2, abs(x-xc)-L/2, abs(x-xc))
    disty = conditional(abs(y-yc)>L/2, abs(y-yc)-L/2, abs(y-yc))
    diffx = x-xc
    diffy = y-yc
    r = conditional(abs(x-xc) <= L/2, sqrt(diffx*diffx + diffy*diffy)/R, sqrt((L-abs(x-xc))**2 + (L-abs(y-yc))**2)/R)
    print(type(r))
    #phi = atan2(x-xc, y - yc)

    rhoc = 1.
    rhob = conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6)

    uth = (1024 * (1.0 - r)**6 * (r)**6)
    #phi = atan2(x-xc, y-yc)
    diffx = conditional(abs(x-xc) <= L/2, x - xc, L-abs(x-xc))
    diffy = conditional(abs(y-yc) <= L/2, y - yc, L-abs(y-yc))
    ux = conditional(r>=1, uc, uc - uth * diffy/(r*R))
    
    uy = conditional(r>=1, vc, vc + uth * diffx/(r*R))
    u0 = as_vector([ux,uy])
    return u0, rhob

#define delx and delt so that delx/delt is constant, since we expect second order 
#convergence in both, delt and delx 
uc = 100.
vc = 100.
L = 10000.
H = 10000.
delx_list = [500., 250., 125., 125/2]
delt_list = [0.0025]#[0.125/16 for i in range(len(delx_list))]
n_timesteps = [100/delt_list[i] for i in range(len(delt_list))]
nx_list = [int(10000/delx_list[i]) for i in range(len(delx_list))]
print("nx_list ", nx_list)
check_freqs= [10 for i in range(len(delx_list))]

#delx_list = [500/(2**i) for i in range(4)]
print("delx_list = ", delx_list)
#delt_list = [0.2*delx_list[i]/uc for i in range(4)]
print("delt_list", delt_list)
#n_timesteps = [100/delt_list[i] for i in range(4)]
print("number time steps = ", n_timesteps)

#check_freqs = [25, 25, 50, 100]

for i in range(len(delx_list)):

    t =0.
    delx=delx_list[i]
    #delx = 5.21e-3
    nx = int(L/delx)
    print("nx = ", nx)
    #delx = L/nx
    print(delx)
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    
    m = PeriodicIntervalMesh(nx, L, distribution_parameters =
                                distribution_parameters)
    mesh = ExtrudedMesh(m, nx, layer_height=delx, periodic=True, name='mesh')
    x, y = SpatialCoordinate(mesh)

    Vc = mesh.coordinates.function_space()

    #Vc = VectorFunctionSpace(mesh, "DG", 2)
    x, y = SpatialCoordinate(mesh)
    xc = L/2
    y_m = 300.
    sigma_m = 500.
    y_s = y_m*exp(-((x - xc)/sigma_m)**2)
    y_expr = y+y_s
    new_coords = Function(Vc).interpolate(as_vector([x,y_expr]))
    mesh.coordinates.assign(new_coords)
    #mesh = Mesh(new_coords)

    #mesh.coordinates.assign(new_coords)
    ##new_coords = Function(Vc).interpolate(as_vector([x,y -(4/H**2) * 0.5*(sin(2*pi*x/L))*((y-H/2)**2 - H**2/4)] ))
    x, y = SpatialCoordinate(mesh)

    xc = L/2 #+ t*uc
    yc = L/2 #+ t*vc

    R = 0.4*L #radius of vortex
    r = sqrt((x - xc)**2 + (y - yc)**2)/R
    phi = atan2(x-xc, y - yc)

    rhoc = 1.
    rhob = conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6)

    uth = (1024 * (1.0 - r)**6 * (r)**6)
    phi = atan2(x-xc, y-yc)
    ux = conditional(r>=1, uc, uc - uth * (y - yc)/(r*R))
    uy = conditional(r>=1, vc, vc + uth * (x - xc)/(r*R))
    u0 = as_vector([ux,uy])


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

    p = 0
    for ip in range(25):
        p += coe[ip] * (r**(12+ip)-1)
    mach = 0.341
    p = 1 + 1024**2 * mach**2 *conditional(r>=1, 0, p)

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

    p = 0
    for ip in range(25):
        p += coe[ip] * (r**(12+ip))
    for ip in range(13):
        p += 2*const_coe[ip] * (1 + 0.5*(1 - r**2)**6)/rhoc * (r**ip)

    p = 11024**2 * rhoc * (uc**2 + vc**2) * p

    p = 1+conditional(r>=1, 0, p)

    '''

    #thetab = thermodynamics_theta(p, rhob)
    #thetab = pref/(R0*rhob)*((p/pref)**(1/gamma))

    R0 = 287.
    gamma = 1.4
    kappa = 2./7.
    pref = Parameters.p_0

    T = p/(rhob*R0)
    thetab = T*(pref/p)**kappa

    Vp = FunctionSpace(mesh, "DG", 1)
    p_out = Function(Vp, name = "p_0").interpolate(p)
    file = File("pinit.pvd")
    pib = thermodynamics_pi(rhob, thetab)
    pib_out = Function(Vp, name = "Pi0").interpolate(pib)
    file.write(p_out, pib_out)

    Problem = compressibleEulerEquations(mesh, vertical_degree=1, horizontal_degree=1)
    Problem.g = 0.
    Problem.H = L # edit later in class
    Problem.u0 = u0
    Problem.rhob = rhob
    Problem.Pib = pib
    Problem.solver_params = sparameters
    Problem.path_out = "../Results/convergence/vortex"+str(nx)
    Problem.thetab = thetab
    Problem.theta_init_pert = 0
    Problem.sponge_fct = False
    Problem.checkpointing = True
    Problem.checkpoint_path = "checkpointVortex"+str(nx)+".h5"
    Problem.check_freq = check_freqs[i]
    Problem.out_error =  True
    #Problem.error_file = "../Results/convergence/vortex"
    Problem.exact_sol = exact_solution

    '''
    n = 32, dt = 0.5, courant = 0.16
    '''
    #dt = 9.7e-4
    dt = delt_list[0]
    
    print("dt = ", dt)


    tmax = 10.
    dumpt = 100*dt
    print ("courant = ", uc*dt/delx)

    Problem.solve(dt=dt, tmax=tmax, dumpt=dumpt, hydrostatic_balance_background=False)

#delx_list = [500.0, 250.0, 125.0, 62.5]
#delt_list [1.0, 0.5, 0.25, 0.125]
#number time steps =  [100.0, 200.0, 400.0, 800.0]

#check_freq = 100 --> idx = 9 for n = 4
#check-freq = 50 --> idx = 9 for n = 3 
#check_freq = 25 --> idx = 9 for n = 2
#check-freq = 25 --> idx = 5 for n =1



