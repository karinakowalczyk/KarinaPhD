from firedrake import *
from .new_spaces import build_spaces
from .physics import *

def compressible_hydrostatic_balance(mesh, parameters, theta0, rho0, lambdar0, pi0=None,
                                     top=False, pi_boundary=Constant(1.0),
                                     water_t=None,
                                     solve_for_rho=False,
                                     horizontal_degree=1,
                                     vertical_degree=1,
                                     params=None):
    """
    Compute a hydrostatically balanced density given a potential temperature
    profile. By default, this uses a vertically-oriented hybridization
    procedure for solving the resulting discrete systems.

    :arg state: The :class:`State` object.
    :arg theta0: :class:`.Function`containing the potential temperature.
    :arg rho0: :class:`.Function` to write the initial density into.
    :arg top: If True, set a boundary condition at the top. Otherwise, set
    it at the bottom.
    :arg pi_boundary: a field or expression to use as boundary data for pi on
    the top or bottom as specified.
    :arg water_t: the initial total water mixing ratio field.
    """

    # Calculate hydrostatic Pi
    #VDG = state.spaces("DG")
    #Vv = state.spaces("Vv")
    #Vtr= state.spaces("Trace")
    _, Vv, Vp, Vt, Vtr = build_spaces(mesh, vertical_degree, horizontal_degree ) # arguments to be set in main function
    W = MixedFunctionSpace((Vv, Vp, Vtr))
    v, pi, lambdar = TrialFunctions(W)
    dv, dpi, gammar = TestFunctions(W)

    n = FacetNormal(mesh)

    # add effect of density of water upon theta
    theta = theta0

    if water_t is not None:
        theta = theta0 / (1 + water_t)

    if top:
        bmeasure = ds_t
        bstring = "bottom"
        vmeasure = ds_b
    else:
        bmeasure = ds_b
        vmeasure = ds_t
        bstring = "top"

    cp = parameters.cp

    alhs = (
        (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
        +cp * dpi*div(theta*v)*dx

        - cp*inner(theta*v, n) * gammar * vmeasure
        - cp*jump(theta*v, n=n) * gammar('+') * (dS_h)

        + cp*inner(theta*dv, n) * lambdar * vmeasure
        + cp*jump(theta*dv, n=n) * lambdar('+') * (dS_h)

        + gammar * lambdar * bmeasure
    )

    arhs = -cp*inner(dv, n)*theta*pi_boundary*bmeasure
    arhs += gammar * pi_boundary*bmeasure

    # Possibly make g vary with spatial coordinates?


    dim = mesh.topological_dimension()
    kvec = [0.0] * dim
    kvec[dim - 1] = 1.0
    k = Constant(kvec)

    g = parameters.g
    arhs -= g*inner(dv, k)*dx

    #bcs = [DirichletBC(W.sub(0), zero(), bstring)]
    bcs =[]

    w = Function(W)
    PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    if params is None:
        #params = {'ksp_type': 'preonly',
         #         'pc_type': 'python',
          #        'mat_type': 'matfree',
           #       'pc_python_type': 'gusto.VerticalHybridizationPC', #EDIT: Use SCPC instead
            #      # Vertical trace system is only coupled vertically in columns
             #     # block ILU is a direct solver!
              #    'vert_hybridization': {'ksp_type': 'preonly',
               #                          'pc_type': 'bjacobi',
                #                         'sub_pc_type': 'ilu'}}

        scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
        params = {"ksp_type": "gmres",
                  "snes_monitor": None,
                  "ksp_monitor": None,
                                  "mat_type": "matfree",
                                  "pc_type": "python",
                                  "pc_python_type": "firedrake.SCPC",
                                  "condensed_field": scpc_parameters,
                                  "pc_sc_eliminate_fields": "0,1"}


    PiSolver = LinearVariationalSolver(PiProblem,
                                       solver_parameters=params,
                                       options_prefix="pisolver")

    PiSolver.solve()

    v, Pi, lambdar = w.split()

    print("Pi max and min = ", Pi.dat.data.max(), Pi.dat.data.min())
    print("theta0 max and min:", theta0.dat.data.max(), theta0.dat.data.min())

    if pi0 is not None:
        pi0.assign(Pi)

    if solve_for_rho:
        w1 = Function(W)
        v, rho, lambdar = w1.split()
        rho.interpolate(thermodynamics_rho(parameters, theta0, Pi))
        print("rho max and min before", rho.dat.data.max(), rho.dat.data.min())

        v, rho, lambdar = split(w1)


        dv, dpi, gammar = TestFunctions(W)
        pi = thermodynamics_pi(parameters, rho, theta0)
        F = (
            (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
            + cp * inner(theta * dv, n) * lambdar * vmeasure
            + cp * jump(theta * dv, n=n) * lambdar('+') * (dS_h)

            - cp * inner(theta*v, n) * gammar * vmeasure
            - cp * jump(theta*v, n=n) * gammar('+') * (dS_h)

            + dpi*div(theta0*v)*dx
            + cp*inner(dv, n)*theta*pi_boundary*bmeasure

            + gammar * (lambdar - pi_boundary) * bmeasure
        )

        F += g*inner(dv, k)*dx
        rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
        rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params,
                                               options_prefix="rhosolver")
        rhosolver.solve()
        v, rho_, lambdar = w1.split()
        rho0.assign(rho_)
        print("rho max", rho0.dat.data.max())
    else:
        rho0.interpolate(thermodynamics_rho(parameters, theta0, Pi))
    lambdar0.assign(lambdar)


# help function
def minimum(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
            static void minify(double *a, double *b) {
                a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
            }
            """, "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]

def applyBC(u):
    bc = DirichletBC(u.function_space(), 0.0, "bottom")
    bc.apply(u)


def compressible_hydrostatic_balance_with_correct_pi_top(mesh, parameters, theta_b, rho_b, lambdar_b, Pi=None,
                                                         vertical_degree=1, horizontal_degree=1):

    
    ## specify solver parameters for the hydrostatic balance
    scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
    piparamsSCPC = {"ksp_type": "gmres",
                    "ksp_monitor": None,
                    #"ksp_view":None,
                    "mat_type": "matfree",
                    #'pc_type':'lu',
                    #'pc_factor_mat_solver_type':'mumps'}
                    "pc_type": "python",
                    "pc_python_type": "firedrake.SCPC",
                    "condensed_field": scpc_parameters,
                    "pc_sc_eliminate_fields": "0,1"}
    
    
    compressible_hydrostatic_balance(mesh, parameters, theta_b, rho_b, lambdar_b, Pi,
                                 top=True, pi_boundary=0.5,
                                 vertical_degree=vertical_degree, horizontal_degree=horizontal_degree,
                                 params=piparamsSCPC)



    # solve with the correct pressure on the top boundary
    p0 = minimum(Pi)

    compressible_hydrostatic_balance(mesh, parameters, theta_b, rho_b, lambdar_b, Pi,
                                    top=True, vertical_degree=vertical_degree, horizontal_degree=horizontal_degree, params=piparamsSCPC)
    p1 = minimum(Pi)
    alpha = 2.*(p1-p0)
    beta = p1-alpha
    pi_top = (1.-beta)/alpha

    print("SOLVE FOR RHO NOW")

    #rho_b to be used later as initial guess for solving Euler equations
    compressible_hydrostatic_balance(mesh, parameters, theta_b, rho_b, lambdar_b, Pi,
                                        top=True, pi_boundary=pi_top, solve_for_rho=True,
                                        vertical_degree=vertical_degree, horizontal_degree=horizontal_degree,
                                        params=piparamsSCPC)

