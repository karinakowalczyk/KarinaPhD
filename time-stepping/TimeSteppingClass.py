from firedrake import * 

class TimeStepping:

    def __init__(self, dtc):
        self.dtc = dtc
        R0 = 6371220.
        H = Constant(5960.)
        Omega = Constant(7.292e-5)  # rotation rate
        #f = 2*Omega*z/Constant(R0)  # Coriolis parameter
        g = Constant(9.8)  # Gravitational constant

        distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

        mesh = IcosahedralSphereMesh(radius=R0,
                                    degree=1,
                                    refinement_level=4,
                                    distribution_parameters = distribution_parameters)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
        n = FacetNormal(mesh)

        # We set up a function space of discontinous bilinear elements for :math:`q`, and
        # a vector-valued continuous function space for our velocity field. ::

        V0 = FunctionSpace(mesh, "CG", degree=3)
        #velocity space
        element = FiniteElement("BDM", triangle, degree=2)
        V1_broken = FunctionSpace(mesh, BrokenElement(element))
        V1 = FunctionSpace(mesh, element)
        #space for height:
        V2 = FunctionSpace(mesh, "DG", 1)
        W_broken = MixedFunctionSpace((V1_broken,V2))
        W = MixedFunctionSpace((V1,V2))

        # We set up the initial velocity field using a simple analytic expression. ::

        # SET UP EXAMPLE

        u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
        u_max = Constant(u_0)
        #solid body rotation (?)
        u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
        file = File('ubar.pvd')
        self.ubar = Function(V1).interpolate(u_expr)

        # define velocity field to be advected:
        x_c = as_vector([1., 0., 0.])
        F_0 = Constant(3.)
        l_0 = Constant(0.25)

        def dist_sphere(x, x_c):
            return acos(dot(x/R0,x_c))


        F_theta = F_0*exp(-dist_sphere(x,x_c)**2/l_0**2)
        #D_expr = conditional(dist_sphere(x,x_c) > 0.5, 0., F_theta)

        D_expr = H - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g

        # Topography.
        b = Function(V2, name="Topography")

        rl = pi/9.0
        lambda_x = atan_2(x[1]/R0, x[0]/R0)
        lambda_c = -pi/2.0
        phi_x = asin(x[2]/R0)
        phi_c = pi/6.0
        minarg = min_value(pow(rl, 2),
                        pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
        bexpr = 2000.0*(1 - sqrt(minarg)/rl)
        b.interpolate(bexpr)
        self.Dn = Function(V2, name = "D").interpolate(D_expr - b)
        self.Dbar = Function(V2).assign(H)

        #un = Function(V1_broken, name = "u").interpolate(velocity)
        self.un = Function(V1, name = "u").assign(self.ubar)
        print("u is set", norm(self.un), norm(self.Dn))

        self.Unp1 = Function(W)
        self.unp1, self.Dnp1 = self.Unp1.subfunctions

        self.Dnp1.assign(self.Dn)
        self.unp1.assign(self.un)

        outward_normals = CellNormal(mesh)
        def perp(u):
            return cross(outward_normals, u)
        
        def compute_vorticity():
            q = TrialFunction(V0)
            p = TestFunction(V0)

            qn = Function(V0, name="Relative Vorticity")
            veqn = q*p*dx + inner(perp(grad(p)), un)*dx
            vprob = LinearVariationalProblem(lhs(veqn), rhs(veqn), qn)
            qparams = {'ksp_type':'cg'}
            qsolver = LinearVariationalSolver(vprob,
                                                solver_parameters=qparams)
            qsolver.solve()

        #to be the solutions, initialised with un, Dn
        self.D = Function(V2).assign(self.Dn)
        self.u = Function(V1_broken).project(self.un)

        def compute_Courant():
            def both(u):
                return 2*avg(u)

            # compute COURANT number
            DG0 = FunctionSpace(mesh, "DG", 0)
            One = Function(DG0).assign(1.0)
            unn = 0.5*(inner(-un, n) + abs(inner(-un, n))) # gives fluxes *into* cell only
            v = TestFunction(DG0)
            Courant_num = Function(DG0, name="Courant numerator")
            Courant_num_form = dt*(both(unn*v)*dS)

            Courant_denom = Function(DG0, name="Courant denominator")
            assemble(One*v*dx, tensor=Courant_denom)
            Courant = Function(DG0, name="Courant")

            assemble(Courant_num_form, tensor=Courant_num)
            courant_frac = Function(DG0).interpolate(Courant_num/Courant_denom)
            Courant.assign(courant_frac)


        # Now we declare our variational forms.  Solving for :math:`\Delta q` at each
        # stage, the explicit timestepping scheme means that the left hand side is just a
        # mass matrix. ::

        dD_trial = TrialFunction(V2)
        phi = TestFunction(V2)
        a_D = phi*dD_trial*dx

        du_trial = TrialFunction(V1_broken)
        w = TestFunction(V1_broken)
        a_u = inner(w, du_trial)*dx


        # The right-hand-side is more interesting.  We define ``n`` to be the built-in
        # ``FacetNormal`` object; a unit normal vector that can be used in integrals over
        # exterior and interior facets.  We next define ``un`` to be an object which is
        # equal to :math:`\vec{u}\cdot\vec{n}` if this is positive, and zero if this is
        # negative.  This will be useful in the upwind terms. ::

        n = FacetNormal(mesh)
        unn = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))

        # We now define our right-hand-side form ``L1`` as :math:`\Delta t` times the
        # sum of four integrals.



        #equations for the advection step
        def eq_D(ubar, Dbar):
            uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
            return (-inner(grad(phi), ubar)*(self.D-Dbar)*dx
                    + jump(phi)*(uup('+')*(self.D-Dbar)('+')
                                    - uup('-')*(self.D-Dbar)('-'))*dS)

        def adv_u(ubar):
            unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

            return(-inner(div(outer(w, ubar)), self.u)*dx
                +dot(jump(w), (unn('+')*self.u('+') - unn('-')*self.u('-')))*dS
            )

        L1_D = dtc*(-eq_D(self.ubar, self.Dbar))

        eq_u = adv_u(self.ubar)
        #adjust to sphere
        unn = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))
        eq_u += unn('+')*inner(w('-'), n('+')+n('-'))*inner(self.u('+'), n('+'))*dS
        eq_u += unn('-')*inner(w('+'), n('+')+n('-'))*inner(self.u('-'), n('-'))*dS

        L1_u = dtc*(-eq_u)
        # In our Runge-Kutta scheme, the first step uses :math:`q^n` to obtain
        # :math:`q^{(1)}`.  We therefore declare similar forms that use :math:`q^{(1)}`
        # to obtain :math:`q^{(2)}`, and :math:`q^{(2)}` to obtain :math:`q^{n+1}`. We
        # make use of UFL's ``replace`` feature to avoid writing out the form repeatedly. ::

        self.D1 = Function(V2); self.D2 = Function(V2)
        L2_D = replace(L1_D, {self.D: self.D1}); L3_D = replace(L1_D, {self.D: self.D2})

        self.u1 = Function(V1_broken); self.u2 = Function(V1_broken)
        L2_u = replace(L1_u, {self.u: self.u1}); L3_u = replace(L1_u, {self.u: self.u2})

        # We now declare a variable to hold the temporary increments at each stage. ::

        self.dD = Function(V2)
        self.du = Function(V1_broken)

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        prob_1_D = LinearVariationalProblem(a_D, L1_D, self.dD)
        self.solv_1_D = LinearVariationalSolver(prob_1_D, solver_parameters=params)
        prob_2_D = LinearVariationalProblem(a_D, L2_D, self.dD)
        self.solv_2_D = LinearVariationalSolver(prob_2_D, solver_parameters=params)
        prob_3_D = LinearVariationalProblem(a_D, L3_D, self.dD)
        self.solv_3_D = LinearVariationalSolver(prob_3_D, solver_parameters=params)

        prob_1_u = LinearVariationalProblem(a_u, L1_u, self.du)
        self.solv_1_u = LinearVariationalSolver(prob_1_u, solver_parameters=params)
        prob_2_u = LinearVariationalProblem(a_u, L2_u, self.du)
        self.solv_2_u = LinearVariationalSolver(prob_2_u, solver_parameters=params)
        prob_3_u = LinearVariationalProblem(a_u, L3_u, self.du)
        self.solv_3_u = LinearVariationalSolver(prob_3_u, solver_parameters=params)

        self.unp1, self.Dnp1 = split(self.Unp1)
        Omega = Constant(7.292e-5)  # rotation rate
        f = 2*Omega*x[2]/Constant(R0)  # Coriolis parameter

        v, rho = TestFunctions(W)
        def proj_u():
            return (-div(v)*g*(self.Dnp1+b)*dx
                    + inner(v, f*perp(self.unp1))*dx 
            )

        def proj_D(Dbar):
            uup = 0.5 * (dot(self.unp1, n) + abs(dot(self.unp1, n)))
            return (-inner(grad(rho), self.unp1)*Dbar*dx
                    + jump(rho)*(uup('+')*Dbar('+')
                                    - uup('-')*Dbar('-'))*dS)

        #make signs consistent

        a_proj_u = inner(v, self.unp1 - self.u)*dx + dtc*proj_u()

        a_proj_D = rho*(self.Dnp1 -self.D)*dx + dtc*proj_D(self.Dbar)

        a_proj = a_proj_u + a_proj_D

        prob = NonlinearVariationalProblem(a_proj, self.Unp1)

        hparams = {
            "snes_lag_jacobian": -2,
            'mat_type': 'matfree',
            'ksp_type': 'gmres',
            #'ksp_monitor': None,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.HybridizationPC',
            'hybridization': {'ksp_type': 'preonly',
                            'pc_type': 'lu'
                            }}

        self.solver_proj = NonlinearVariationalSolver(prob, solver_parameters=hparams)


        # We now run the time loop.  This consists of three Runge-Kutta stages, and every
        # 20 steps we write out the solution to file and print the current time to the
        # terminal. ::

        
        self.unp1, self.Dnp1 = self.Unp1.subfunctions
    
    
    def advection_equations(self):

        dD_trial = TrialFunction(V2)
        phi = TestFunction(V2)
        a_D = phi*dD_trial*dx

        du_trial = TrialFunction(V1_broken)
        w = TestFunction(V1_broken)
        a_u = inner(w, du_trial)*dx

        n = FacetNormal(mesh)
        unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        # We now define our right-hand-side form ``L1`` as :math:`\Delta t` times the
        # sum of four integrals.


        #equations for the advection step
        def eq_D(ubar, Dbar):
            uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
            return (-inner(grad(phi), ubar)*(D-Dbar)*dx
                    + jump(phi)*(uup('+')*(D-Dbar)('+')
                                    - uup('-')*(D-Dbar)('-'))*dS)

        def adv_u(ubar):
            unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

            return(-inner(div(outer(w, ubar)), u)*dx
                +dot(jump(w), (unn('+')*u('+') - unn('-')*u('-')))*dS
            )
        L1_D = dtc*(-eq_D(ubar, Dbar))

        eq_u = adv_u(self.ubar)
        #adjust to sphere
        unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        eq_u += unn('+')*inner(w('-'), n('+')+n('-'))*inner(u('+'), n('+'))*dS
        eq_u += unn('-')*inner(w('+'), n('+')+n('-'))*inner(u('-'), n('-'))*dS

        L1_u = dtc*(-eq_u)

        return(a_u, a_D, L1_u, L1_D)
    
    def advection_solvers_u(self, du):

        # We now declare a variable to hold the temporary increments at each stage. ::

        du = Function(V1_broken)

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
        prob_1_u = LinearVariationalProblem(a_u, L1_u, du)
        solv_1_u = LinearVariationalSolver(prob_1_u, solver_parameters=params)
        prob_2_u = LinearVariationalProblem(a_u, L2_u, du)
        solv_2_u = LinearVariationalSolver(prob_2_u, solver_parameters=params)
        prob_3_u = LinearVariationalProblem(a_u, L3_u, du)
        solv_3_u = LinearVariationalSolver(prob_3_u, solver_parameters=params)

        return (solv_1_u, solv_2_u, solv_3_u)
        
    
    def advection_solvers_D(dD):

        D1 = Function(V2); D2 = Function(V2)
        L2_D = replace(L1_D, {D: D1}); L3_D = replace(L1_D, {D: D2})

        # We now declare a variable to hold the temporary increments at each stage. ::

        dD = Function(V2)

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        prob_1_D = LinearVariationalProblem(a_D, L1_D, dD)
        solv_1_D = LinearVariationalSolver(prob_1_D, solver_parameters=params)
        prob_2_D = LinearVariationalProblem(a_D, L2_D, dD)
        solv_2_D = LinearVariationalSolver(prob_2_D, solver_parameters=params)
        prob_3_D = LinearVariationalProblem(a_D, L3_D, dD)
        solv_3_D = LinearVariationalSolver(prob_3_D, solver_parameters=params)

        return (solv_1_D, solv_2_D, solv_3_D)

            
    def advection_SSPRK3(self,):

        self.u.project(self.un)
        self.D.assign(self.Dn)

        self.solv_1_D.solve()
        self.D1.assign(self.D + self.dD)

        self.solv_2_D.solve()
        self.D2.assign(0.75*self.D + 0.25*(self.D1 + self.dD))

        self.solv_3_D.solve()
        self.D.assign((1.0/3.0)*self.D + (2.0/3.0)*(self.D2 + self.dD))

        self.solv_1_u.solve()
        self.u1.assign(self.u + self.du)

        self.solv_2_u.solve()
        self.u2.assign(0.75*self.u + 0.25*(self.u1 + self.du))

        self.solv_3_u.solve()
        self.u.assign((1.0/3.0)*self.u + (2.0/3.0)*(self.u2 + self.du))

    def projection_step(self):
         # PROJECTION STEP
        self.Dnp1.assign(self.D)
        self.unp1.project(self.u) #u from discontinuous space

        self.solver_proj.solve()

        self.Dn.assign(self.Dnp1)
        self.un.assign(self.unp1)
        self.ubar.assign(self.un)
    
