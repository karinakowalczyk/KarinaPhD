from firedrake import * 
import timeit
from firedrake.petsc import PETSc

class SWEWithProjection:

    def __init__(self, mesh, dtc, u_expr, D_expr, H, bexpr = None, 
                 second_order = False, u_exact = None, D_exact = None, 
                 n_adv_cycles = 1, slate_inv = False):


        self.dtc = dtc
        self.n_adv_cycles = n_adv_cycles
        self.mesh = mesh
        x = SpatialCoordinate(mesh)
        self.second_order = second_order
        self.slate_inv = slate_inv
        R0 = 6371220.
        Omega = Constant(7.292e-5)  # rotation rate
        f = 2*Omega*x[2]/Constant(R0)  # Coriolis parameter
        g = Constant(9.8)  # Gravitational constant
        n = FacetNormal(mesh)
        
        # set up function spaces
        self.V0 = FunctionSpace(mesh, "CG", degree=3)
        #velocity space
        element = FiniteElement("BDM", triangle, degree=2)
        V1_broken = FunctionSpace(mesh, BrokenElement(element))
        V1 = FunctionSpace(mesh, element)
        #space for height:
        V2 = FunctionSpace(mesh, "DG", 1)
        W = MixedFunctionSpace((V1,V2))

        self.ubar = Function(V1).interpolate(u_expr)
        self.u0 = Function(V1).assign(self.ubar)
        
        self.u_exact = None
        self.D_exact = None


        # Topography.
        if bexpr:
            b = Function(V2, name="Topography")
            b.interpolate(bexpr)
        else:
            b = 0
        self.b = b
        self.Dn = Function(V2, name = "D").interpolate(D_expr - b)
        self.D0 = Function(V2, name = "D0").assign(self.Dn)
        #H_avg = assemble(D_expr*dx)
        self.Dbar = Function(V2).assign(H)
        self.Dplusb = Function(V2, name = "D+b").assign(self.Dn + self.b)

        #un = Function(V1_broken, name = "u").interpolate(velocity)
        self.un = Function(V1, name = "u").assign(self.ubar)
        self.u0 = Function(V1, name = "u0").assign(self.un)
        print("u is set", norm(self.un), norm(self.Dn))
        self.u_pert = Function(V1, name = 'delta u').assign(self.un - self.u0)
        self.D_pert = Function(V2, name = 'delta D').assign(self.Dn - self.D0)


        self.Unp1 = Function(W)
        self.unp1, self.Dnp1 = self.Unp1.subfunctions

        if second_order:
            self.Uhat = Function(W)
            self.uhat, self.Dhat = self.Uhat.subfunctions

        self.Dnp1.assign(self.Dn)
        self.unp1.assign(self.un)

        if second_order:
             self.unph = Function(V1)

        outward_normals = CellNormal(mesh)
        def perp(u):
            return cross(outward_normals, u)
        
        # set up courant number
        print("compute Courant number")
        def both(u):
            return 2*avg(u)

        self.DG0 = FunctionSpace(self.mesh, "DG", 0)
        One = Function(self.DG0).assign(1.0)
        n = FacetNormal(self.mesh)
        unn = 0.5*(inner(-self.un, n) + abs(inner(-self.un, n))) # gives fluxes *into* cell only
        v = TestFunction(self.DG0)
        self.Courant_num = Function(self.DG0, name="Courant numerator")
        self.Courant_num_form = self.dtc*(both(unn*v)*dS)

        self.Courant_denom = Function(self.DG0, name="Courant denominator")
        assemble(One*v*dx, tensor=self.Courant_denom)
        self.Courant = Function(self.DG0, name="Courant")

        assemble(self.Courant_num_form, tensor=self.Courant_num)
        courant_frac = Function(self.DG0).interpolate(self.Courant_num/self.Courant_denom)
        self.Courant.assign(courant_frac)
        
        
        # VORTICITY
      
        q = TrialFunction(self.V0)
        p = TestFunction(self.V0)

        self.qn = Function(self.V0, name="relative vorticity")
        self.pot_qn = Function(self.V0, name="potential vorticity")
        veqn = q*p*dx + inner(perp(grad(p)), self.un)*dx
        vprob = LinearVariationalProblem(lhs(veqn), rhs(veqn), self.qn)
        qparams = {'ksp_type':'cg'}
        self.qsolver = LinearVariationalSolver(vprob, solver_parameters=qparams)
        self.qsolver.solve()
        pot_vort_eqn = (self.Dn*q*p*dx + inner(perp(grad(p)), self.un)*dx - f*p*dx)
        pot_vort_prob = LinearVariationalProblem(lhs(pot_vort_eqn), rhs(pot_vort_eqn), self.pot_qn)
        self.pot_vort_solver = LinearVariationalSolver(pot_vort_prob, solver_parameters=qparams)
        self.pot_vort_solver.solve()

        self.enstrophy0 = assemble(0.5*self.pot_qn*self.pot_qn*self.Dn*dx)
        self.energy0 = assemble((0.5*inner(self.u0,self.u0)*self.D0 + 0.5 * g * (self.D0+self.b)**2)*dx) 
        self.div_l20 = sqrt(assemble(div(self.u0)*div(self.u0)*dx))
        
        #to be the solutions, initialised with un, Dn
        self.D = Function(V2).assign(self.Dn)
        self.u = Function(V1_broken).project(self.un)


        # Now we declare our variational forms.  Solving for :math:`\Delta q` at each
        # stage, the explicit timestepping scheme means that the left hand side is just a
        # mass matrix. ::

        dD_trial = TrialFunction(V2)
        phi = TestFunction(V2)
        a_D = phi*dD_trial*dx
        a_D_slate = Tensor(a_D).inv
        self.dmass_inv = assemble(a_D_slate)

        du_trial = TrialFunction(V1_broken)
        w = TestFunction(V1_broken)
        a_u = inner(w, du_trial)*dx
        a_u_slate = Tensor(a_u).inv
        self.umass_inv = assemble(a_u_slate)


        # The right-hand-side is more interesting.  We define ``n`` to be the built-in
        # ``FacetNormal`` object; a unit normal vector that can be used in integrals over
        # exterior and interior facets.  We next define ``unn`` to be an object which is
        # equal to :math:`\vec{u}\cdot\vec{n}` if this is positive, and zero if this is
        # negative.  This will be useful in the upwind terms. ::

        n = FacetNormal(mesh)
        unn = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))


        # For the advection step, we now define our right-hand-side form ``L1`` as :math:`\Delta t` times the
        # sum of four integrals.
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

        L1_D = (dtc/n_adv_cycles)*(-eq_D(self.ubar, self.Dbar))
        
        eq_u = adv_u(self.ubar)
        #adjust to sphere
        eq_u += unn('+')*inner(w('-'), n('+')+n('-'))*inner(self.u('+'), n('+'))*dS
        eq_u += unn('-')*inner(w('+'), n('+')+n('-'))*inner(self.u('-'), n('-'))*dS

        L1_u = (dtc/n_adv_cycles)*(-eq_u)


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

        if slate_inv:
            self.L1_D = L1_D
            self.L2_D = L2_D
            self.L3_D = L3_D
            self.rhs = Cofunction(V2.dual())

            self.L1_u = L1_u
            self.L2_u = L2_u
            self.L3_u = L3_u
            self.rhs_u = Cofunction(V1_broken.dual())


        

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu', }
        prob_1_D = LinearVariationalProblem(a_D, L1_D, self.dD, constant_jacobian=True)
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
        

        v, rho = TestFunctions(W)
        def proj_u():
            return (-div(v)*g*(self.Dnp1+b)*dx
                    + inner(v, f*perp(self.unp1))*dx 
            )

        def proj_D(Dbar):
            return rho* div(self.unp1*Dbar)*dx

        factor = Constant(1.)
        if self.second_order:
             factor = Constant(0.5)
            
        
        a_proj_u = inner(v, self.unp1 - self.u)*dx + (dtc*factor)*proj_u()
             
        a_proj_D = rho*(self.Dnp1 -self.D)*dx + (dtc*factor)*proj_D(self.Dbar)

        a_proj = a_proj_u + a_proj_D

        prob = NonlinearVariationalProblem(a_proj, self.Unp1)

        hparams = {
            "snes_lag_jacobian": -2,
            'mat_type': 'matfree',
            'ksp_type': 'gmres',
            'ksp_converged_reason': None,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.HybridizationPC',
            'hybridization': {'ksp_type': 'preonly',
                            'pc_type': 'lu'
                            }}

        self.solver_proj = NonlinearVariationalSolver(prob, solver_parameters=hparams)

        self.projector_broken = Projector(self.un, self.u, solver_parameters = params)

        
        if second_order:

            self.uhat, self.Dhat = split(self.Uhat)
            self.projector_adv = Projector(self.uhat, self.u, solver_parameters = params)
         
            def second_order_first_eq_u():
                return (-div(v)*g*(self.Dn+b)*dx
                        + inner(v, f*perp(self.un))*dx 
                )

            def second_order_first_eq_D(Dbar):
                #uup = 0.5 * (dot(self.un, n) + abs(dot(self.un, n)))
                #return (-inner(grad(rho), self.un)*Dbar*dx
                #        + jump(rho)*(uup('+')*Dbar('+')
                #                        - uup('-')*Dbar('-'))*dS)
                return rho* div(self.un*Dbar)*dx

            a_second_order_u = inner(v, self.uhat - self.un)*dx + (dtc/2)*second_order_first_eq_u()

            a_second_order_D = rho*(self.Dhat -self.Dn)*dx + (dtc/2)*second_order_first_eq_D(self.Dbar)
            a_second_order_1 = a_second_order_u+a_second_order_D
            prob_2_order_1 = NonlinearVariationalProblem(a_second_order_1, self.Uhat)

            params2 = {"snes_lag_jacobian": -2,'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu', }
            self.solver_2ndorder_1st = NonlinearVariationalSolver(prob_2_order_1, solver_parameters=params2)
        
        self.unp1, self.Dnp1 = self.Unp1.subfunctions

        if self.second_order:
            self.uhat, self.Dhat = self.Uhat.subfunctions

    @PETSc.Log.EventDecorator("second_order_1st_step")  
    def second_order_1st_step(self,):
         self.uhat.assign(self.un)
         self.Dhat.assign(self.Dn)
         self.solver_2ndorder_1st.solve()
       

    def advection_SSPRK3_single_step(self,):

        if self.slate_inv:
            print("using slate mass inverse")
            assemble(self.L1_D, tensor = self.rhs)
            with self.dD.dat.vec_wo as x_, self.rhs.dat.vec_ro as b_:
                self.dmass_inv.petscmat.mult(b_, x_)
        else:
            self.solv_1_D.solve()
        self.D1.assign(self.D + self.dD)

        if self.slate_inv:
            print("using slate mass inverse")
            assemble(self.L2_D, tensor = self.rhs)
            with self.dD.dat.vec_wo as x_, self.rhs.dat.vec_ro as b_:
                self.dmass_inv.petscmat.mult(b_, x_)
        else:
            self.solv_2_D.solve()
        self.D2.assign(0.75*self.D + 0.25*(self.D1 + self.dD))

        if self.slate_inv:
            print("using slate mass inverse")
            assemble(self.L3_D, tensor = self.rhs)
            with self.dD.dat.vec_wo as x_, self.rhs.dat.vec_ro as b_:
                self.dmass_inv.petscmat.mult(b_, x_)
        else:
            self.solv_3_D.solve()
        self.D.assign((1.0/3.0)*self.D + (2.0/3.0)*(self.D2 + self.dD))

        if self.slate_inv:
            print("using slate mass inverse for u")
            assemble(self.L1_u, tensor = self.rhs_u)
            with self.du.dat.vec_wo as x_, self.rhs_u.dat.vec_ro as b_:
                self.umass_inv.petscmat.mult(b_, x_)
        else:
            self.solv_1_u.solve()
        self.u1.assign(self.u + self.du)

        if self.slate_inv:
            print("using slate mass inverse for u")
            assemble(self.L2_u, tensor = self.rhs_u)
            with self.du.dat.vec_wo as x_, self.rhs_u.dat.vec_ro as b_:
                self.umass_inv.petscmat.mult(b_, x_)
        else:
            self.solv_2_u.solve()
        self.u2.assign(0.75*self.u + 0.25*(self.u1 + self.du))

        if self.slate_inv:
            print("using slate mass inverse for u")
            assemble(self.L3_u, tensor = self.rhs_u)
            with self.du.dat.vec_wo as x_, self.rhs_u.dat.vec_ro as b_:
                self.umass_inv.petscmat.mult(b_, x_)
        else:
            self.solv_3_u.solve()
        self.u.assign((1.0/3.0)*self.u + (2.0/3.0)*(self.u2 + self.du))

    @PETSc.Log.EventDecorator("advection_step")     
    def advection_SSPRK3(self,):

        if self.second_order: 
            self.projector_adv.project()
            #self.u.project(self.uhat)
            self.D.assign(self.Dhat)
        else:
            self.projector_broken.project()
            #self.u.project(self.un)
            self.D.assign(self.Dn)

        for i in range(self.n_adv_cycles):
            print("adv. subscyle number ", i+1)
            self.advection_SSPRK3_single_step()


    @PETSc.Log.EventDecorator("projection_step")
    def projection_step(self):

    
        #self.Dnp1.assign(self.D)
        #self.unp1.project(self.u) #Todo: assign(self.un) or nothing, u from discontinuous space

        self.solver_proj.solve()

        self.Dn.assign(self.Dnp1)
        self.un.assign(self.unp1)
        self.u_pert.assign(self.un - self.u0)
        self.D_pert.assign(self.Dn - self.D0)
        self.ubar.assign(self.un)
        self.Dplusb.assign(self.Dn + self.b)
    
    
    @PETSc.Log.EventDecorator("compute_vorticity")    
    def compute_vorticity(self):
            '''
            outward_normals = CellNormal(self.mesh)

            def perp(u):
                return cross(outward_normals, u)
            
            q = TrialFunction(self.V0)
            p = TestFunction(self.V0)

            qn = Function(self.V0, name="Relative Vorticity")
            veqn = q*p*dx + inner(perp(grad(p)), self.un)*dx
            vprob = LinearVariationalProblem(lhs(veqn), rhs(veqn), qn)
            qparams = {'ksp_type':'cg'}
            qsolver = LinearVariationalSolver(vprob, solver_parameters=qparams)
            '''
            self.qsolver.solve()
            self.pot_vort_solver.solve()
            #return qn

    @PETSc.Log.EventDecorator("compute_courant")    
    def compute_Courant(self):
            '''
            print("compute Courant number")
            
            def both(u):
                return 2*avg(u)

            DG0 = FunctionSpace(self.mesh, "DG", 0)
            One = Function(DG0).assign(1.0)
            n = FacetNormal(self.mesh)
            unn = 0.5*(inner(-self.un, n) + abs(inner(-self.un, n))) # gives fluxes *into* cell only
            v = TestFunction(DG0)
            Courant_num = Function(DG0, name="Courant numerator")
            Courant_num_form = self.dtc*(both(unn*v)*dS)
            

            Courant_denom = Function(DG0, name="Courant denominator")
            assemble(One*v*dx, tensor=Courant_denom)
            Courant = Function(DG0, name="Courant")
            '''

            assemble(self.Courant_num_form, tensor=self.Courant_num)
            courant_frac = Function(self.DG0).interpolate(self.Courant_num/self.Courant_denom)
            self.Courant.assign(courant_frac)
            #return Courant

    @PETSc.Log.EventDecorator("compute_energies")    
    def compute_phys_quant(self):
         g = Constant(9.8)  # Gravitational constant
         energy = assemble((0.5*inner(self.un,self.un)*self.Dn + 0.5 * g * (self.Dn+self.b)**2)*dx) # this is not the exact energy, there is an additional term, which is conserved anyways
         enstrophy = assemble(0.5*self.pot_qn*self.pot_qn*self.Dn*dx)
         div_l2 = sqrt(assemble(div(self.un)*div(self.un)*dx))

         #energy_error = abs(energy - self.energy0)/self.energy0
         #enstr_error = abs(enstrophy - self.enstrophy0)/self.enstrophy0
         #div_l2_error = abs(div_l2 - self.div_l20)/self.div_l20
         return energy, enstrophy, div_l2 #, energy_error, enstr_error, div_l2_error

    
    def compute_error_sbr(self):
        # EXACT SOLUTION
        u_error = sqrt(assemble(inner(self.un - self.u_exact, self.un - self.u_exact)*dx))/sqrt(assemble(inner(self.u_exact, self.u_exact)*dx))
        D_error = sqrt(assemble(inner(self.Dn - self.D_exact, self.Dn - self.D_exact)*dx))/sqrt(assemble(inner(self.D_exact, self.D_exact)*dx))
        return u_error, D_error
        
