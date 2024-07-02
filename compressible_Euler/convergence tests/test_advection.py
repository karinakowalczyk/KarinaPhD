from firedrake import *
import sympy as sp
import matplotlib.pyplot as plt
from tools.physics import * 
from tools.new_spaces import *
from tools.compressible_Euler import *
import numpy as np

'''
testing the equation 
u_t + v dot grad u = 0
for the vortex with v = (1,1)
'''


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
    distx = conditional(abs(x-xc)>L/2, L-abs(x-xc), abs(x-xc))
    disty = conditional(abs(y-yc)>L/2, L-abs(y-yc), abs(y-yc))
    diffx = x-xc
    diffy = y-yc
    r = sqrt(distx*distx + disty*disty)/R
    #r = conditional(abs(x-xc) <= L/2, sqrt(diffx*diffx + diffy*diffy)/R, sqrt((L-abs(x-xc))**2 + (L-abs(y-yc))**2)/R)
    print(type(r))
    #phi = atan2(x-xc, y - yc)

    rhoc = 1.
    rhob = conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6)

    uth = (1024 * (1.0 - r)**6 * (r)**6)
    #phi = atan2(x-xc, y-yc)
    #diffx = conditional(abs(x-xc) <= L/2, x - xc, L-abs(x-xc))
    #diffy = conditional(abs(y-yc) <= L/2, y - yc, L-abs(y-yc))
    signx = conditional(x>xc,-1, +1 )
    signy = conditional(y>yc,-1, +1 )
    diffx = conditional(abs(x-xc) <= L/2, x - xc, signx*(L-abs(x-xc)))
    diffy = conditional(abs(y-yc) <= L/2, y - yc, signy*(L-abs(y-yc)))
    ux = conditional(r>=1, 0, -uth * diffy/(r*R))
    
    uy = conditional(r>=1, 0, uth * diffx/(r*R))
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
    R0 = 287.
    gamma = 1.4
    kappa = 2./7.
    pref = Parameters.p_0
    T = p/(rhob*R0)
    thetab = T*(pref/p)**kappa
    return u0, rhob, thetab

#define delx and delt so that delx/delt is constant, since we expect second order 
#convergence in both, delt and delx 
uc = 100.
vc = 100.
L = 10000.
H = 10000.
delx_list = [1000., 500., 250., 125., 125./2]
dt = 0.0025#[0.125/16 for i in range(len(delx_list))]

nx_list = [int(10000/delx_list[i]) for i in range(len(delx_list))]
print("nx_list ", nx_list)



def project_constraint_space(u0, V, T):

    """
    apply boundary condition u dot n = 0 to the initial velocity
    in the velocity FEM space V, where V is  the new broken space and
    T the trace space,
    project by solving the according PDE
    """

    W = V*T
    w, mu = TestFunctions(W)
    U = Function(W)
    u, lambdar = split(U)
    n = FacetNormal(V.mesh())

    a = (inner(w, u)*dx - inner(w, u0)*dx
         + jump(w, n)*lambdar('+')*dS_h
         + inner(w, n)*lambdar*ds_tb
         + jump(u, n)*mu('+')*dS_h
         + inner(u, n)*mu*ds_tb
         )
    sparameters_exact = {"mat_type": "aij",
                         'snes_monitor': None,
                         # 'snes_stol': 1e-50,
                         # 'snes_view': None,
                         # 'snes_type' : 'ksponly',
                         'ksp_monitor_true_residual': None,
                         'snes_converged_reason': None,
                         'ksp_converged_reason': None,
                         "ksp_type": "preonly",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}

    problem = NonlinearVariationalProblem(a, U)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sparameters_exact)

    solver.solve()

    return U

errors = []




for i in range(len(delx_list)):
    dT = Constant(1.)
    t =0.
    delx=delx_list[i]
    nx = int(L/delx)
    print("nx = ", nx)
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

    V, _, _, _, _ = build_spaces(mesh, 1, 1)
  
    n = FacetNormal(mesh)
    un = Function(V, name = "Un")
    unp1 = Function(V, name = "Unp1")
    

    unph = 0.5*(un + unp1)
    unph_mean = 0.5*(unph("+") + unph("-"))
    dS = dS_h + dS_v
    v = Constant(as_vector([100.,100.]))
    v_mean = 0.5*(v('+') + v('-'))

    def unn(r):
        return 0.5*(
                    dot(v_mean, n(r))
                    + abs(dot(v_mean, n(r)))
                    )
    def uadv_eq(w):
        return (-inner(div(outer(w, v)), unph)*dx 
                + dot(jump(w),(unn('+')*unph('+') - unn('-')*unph('-')))*dS)
    
    w = TestFunction(V)
    eqn = (inner(w, unp1 - un)*dx + dT * (
                    uadv_eq(w)))
    nprob = NonlinearVariationalProblem(eqn, unp1)
    solver = NonlinearVariationalSolver(nprob, solver_parameters=sparameters_star)
    
    u0, _, _ = exact_solution(x,y,t)
    un.project(u0)

    uerror = norm(un-u0)
    with open("test_adv"+str(nx)+"_uerrors.txt", 'w') as file_uerr:
                file_uerr.write(str(uerror)+'\n')

    file = File("test_adv"+str(nx)+".pvd")
    u_exact_diff = Function(V, name = 'u_diff').project(un - u0)
    file.write(un, u_exact_diff)

    tdump = 0.
    dumpt = 100*dt
    t = 0.
    step = 0
    tmax = 10.
    dT.assign(dt)
       

    PETSc.Sys.Print('tmax', tmax, 'dt', dt)
   

    while t < tmax - 0.5*dt:
        step+=1
        t += dt
        PETSc.Sys.Print(t)
        tdump += dt

        solver.solve()

        un.assign(unp1)
        if tdump > dumpt - dt*0.5:
            u_exact, _, _ = exact_solution(x,y,t)
            uerror = norm(un-u_exact)
            with open("test_adv"+str(nx)+"_uerrors.txt", 'a') as file_uerr:
                file_uerr.write(str(uerror)+'\n')
            #u_exact_diff.project(un - u_exact)
            file.write(un, u_exact_diff)

            tdump -= dumpt
            PETSc.Sys.Print(solver.snes.getIterationNumber())
        

   
print("number time steps", step)
'''

 U_proj = project_constraint_space(u0, V, T)
    u_proj, _ = U_proj.subfunctions 
    file = File("test"+str(nx)+".pvd")
    file.write(u_proj)
    error = norm(u0-u_proj)/norm(u0)
    errors.append(error)
    print(errors)

fig, axes = plt.subplots()

mesh_sizes = np.array([1/10, 1/20, 1/40, 1/80, 1/160])
hpower = mesh_sizes**2
axes.set_title("Loglog plot of velocity errors")
plt.loglog(mesh_sizes, errors, color = "red", label = "u-error", marker = ".")
plt.loglog(mesh_sizes, 1e-2*hpower, color = "orange", label = "h^2", marker = ".")
axes.legend()
fig.savefig("uerrors.png")
plt.show()
'''