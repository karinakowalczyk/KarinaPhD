from firedrake import *
#mesh parameters

L = 100000.
H = 30000.  # Height position of the model top
delx = 500*2
delz = 250*2
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns

# build mesh
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = FacetNormal(mesh)

degree = 0
#family = "BDFM"
family = "RT"
#velocity space
element = BrokenElement(FiniteElement(family, quadrilateral, degree+1))
V1 = VectorFunctionSpace(mesh, element)
#space for height:
V2 = FunctionSpace(mesh, "DG", degree)
W = MixedFunctionSpace((V1,V2))

#set up test and trial functions
U_star = Function(W)
un=Function(V1).interpolate(as_vector([0.0,0.0]))
Dn=Function(V2).interpolate(Constant(0.))
u_star, D_star = split(U_star)

w, phi = TestFunctions(W)

#set up equations
uup = lambda u: 0.5 * (dot(u, n) + abs(dot(u, n)))

def perp_u_upwind(u):
    Upwind = 0.5 * (sign(dot(u, n)) + 1)
    return Upwind('+')*perp(u('+')) + Upwind('-')*perp(u('-'))

def adv_eq(u, ubar, w):
    return (
        + -inner(perp(grad(inner(w, perp(ubar)))), u)*dx
        - inner(jump(inner(w, perp(ubar)), n), perp_u_upwind(u))*(dS_v+dS_h)
        - inner(ubar, u) * div(w) * dx
        )

def eq_D(D, Dbar, ubar, phi):
    h = D-Dbar
    uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
    return (- inner(grad(phi), ubar)*h*dx
            + jump(phi)*(uup('+')*h('+')
                            - uup('-')*h('-'))*(dS_v+dS_h))
ubar = Function(V1).assign(un)
Dbar = Dn
delT = 1.

eqn = inner(w, u_star - un)*dx + delT*adv_eq(u_star, ubar, w) + inner(phi, D_star - Dn)*dx + delT*eq_D(D_star, Dbar, ubar, phi)
prob = NonlinearVariationalProblem(eqn, U_star)

sparameters_exact = { "mat_type": "aij",
                   'snes_monitor': None,
                   'snes_stol': 1e-50,
                   #'snes_view': None,
                   #'snes_type' : 'ksponly',
                   'ksp_monitor_true_residual': None,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None,
                   "ksp_type" : "preonly",
                   "pc_type" : "ilu",
                   #"pc_factor_mat_solver_type": "mumps"
                   }
solver = NonlinearVariationalSolver(prob, solver_parameters=sparameters_exact)

#just one time-step
solver.solve()