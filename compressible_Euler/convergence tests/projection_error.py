from firedrake import *
import sympy as sp
import matplotlib.pyplot as plt
from tools.physics import * 
from tools.new_spaces import *
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
delx_list = [1000., 500., 250., 125., 125./2]
delt_list = [0.25]#[0.125/16 for i in range(len(delx_list))]
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

    V, _, _, _, T = build_spaces(mesh, 1, 1)

    u0, _ = exact_solution(x,y,t)

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
