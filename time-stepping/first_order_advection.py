from firedrake import *
#mesh parameters

def RT_Element_broken_ext(mesh, horizontal_degree, vertical_degree):
    '''
        mesh: expected to be an extruded mesh
    '''
    cell = mesh._base_mesh.ufl_cell().cellname()
    S1 = FiniteElement("CG", cell, horizontal_degree + 1)  # EDIT: family replaced by CG (was called with RT before)
    S2 = FiniteElement("DG", cell, horizontal_degree, variant="equispaced")

    # vertical base spaces
    T0 = FiniteElement("CG", interval, vertical_degree + 1, variant="equispaced")
    T1 = FiniteElement("DG", interval, vertical_degree, variant="equispaced")

    # build spaces
    V2h_elt_broken =BrokenElement(HDiv(TensorProductElement(S1, T1)))
    V2t_elt = TensorProductElement(S2, T0)
    #V3_elt = TensorProductElement(S2, T1)
    V2v_elt_broken = BrokenElement(HDiv(V2t_elt))
    V2_elt = V2h_elt_broken + V2v_elt_broken
    
    return V2_elt
   

#L = 100000.
#H = 30000.  # Height position of the model top
#delx = 500*2
#delz = 250*2
#nlayers = H/delz  # horizontal layers
#columns = L/delx  # number of columns

# build mesh
#m = PeriodicIntervalMesh(columns, L)
#mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
mesh = UnitSquareMesh(40,40)

n = FacetNormal(mesh)

#velocity space
element = FiniteElement("BDM", triangle, degree=2)
V1_broken = FunctionSpace(mesh, BrokenElement(element))
V1 = FunctionSpace(mesh, element)
#space for height:
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1_broken,V2))

#set up test and trial functions
U_star = Function(W)
u_star, D_star = split(U_star)
un=Function(V1).interpolate(as_vector([0.0,0.0]))
Dn=Function(V2).interpolate(Constant(0.))

w, phi = TestFunctions(W)

#set up equations
uup = lambda u: 0.5 * (dot(u, n) + abs(dot(u, n)))

def perp_u_upwind(u):
    Upwind = 0.5 * (sign(dot(u, n)) + 1)
    return Upwind('+')*perp(u('+')) + Upwind('-')*perp(u('-'))

def adv_eq(u, ubar, w):
    return (
        + -inner(perp(grad(inner(w, perp(ubar)))), u)*dx
        - inner(jump(inner(w, perp(ubar)), n), perp_u_upwind(u))*(dS)
        - inner(ubar, u) * div(w) * dx
        )

def eq_D(D, Dbar, ubar, phi):
    h = D-Dbar
    uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
    return (- inner(grad(phi), ubar)*h*dx
            + jump(phi)*(uup('+')*h('+')
                            - uup('-')*h('-'))*(dS))

ubar = Function(V1).assign(un)
Dbar = Dn
delT = 1.

eqn = inner(w, u_star - un)*dx  + delT*adv_eq(u_star, ubar, w) + inner(phi, D_star - Dn)*dx + delT*eq_D(D_star, Dbar, ubar, phi)
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

