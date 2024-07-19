from ufl import TestFunctions
from firedrake import *


class Parameters:
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    g = 9.810616


def thermodynamics_rho(theta_v, pi):
    """
    Returns an expression for the dry density rho in kg / m^3
    from the (virtual) potential temperature and Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg theta_v: the virtual potential temperature in K.
    :arg pi: the Exner pressure.
    """

    kappa = Parameters.kappa
    p_0 = Parameters.p_0
    R_d = Parameters.R_d

    return p_0 * pi ** (1 / kappa - 1) / (R_d * theta_v)


def thermodynamics_pi(rho, theta_v):
    """
    Returns an expression for the Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density of air in kg / m^3.
    :arg theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    kappa = Parameters.kappa
    p_0 = Parameters.p_0
    R_d = Parameters.R_d

    return (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa))



def thermodynamics_theta(p, rho):
    R = 287.
    gamma = 1.4
    pref = Parameters.p_0
    T = p/rho*(1/R)
    theta = T*(pref/p)**()
    return theta


def apply_BC_def_mesh(u0, V, T):

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
