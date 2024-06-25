from firedrake import (SpatialCoordinate, Function, as_vector,
                       Constant, FacetNormal, split,
                       TestFunctions,
                       TestFunction,
                       FunctionSpace,
                       conditional, sin,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver,
                       DirichletBC,
                       dx, dS_h, dS_v, ds_t, ds_b, ds_tb,
                       dot, sign, inner, grad, norm, outer, 
                       curl, cross, jump, perp, div,
                       File, CheckpointFile,
                       assemble
                       )
from tools import (Parameters, build_spaces, build_spaces_slice_3D,
                   compressible_hydrostatic_balance_with_correct_pi_top,
                   thermodynamics_pi, apply_BC_def_mesh
                   )


import numpy as np
from petsc4py import PETSc


class compressibleEulerEquations:

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1,
                 slice_3D=False, mesh_periodic=True, rhs = False):

        self.mesh = mesh
        self.vertical_degree = vertical_degree
        self.horizontal_degree = horizontal_degree
        self.H = None
        self.u0 = Constant(as_vector([0,0]))
        self. dT = Constant(5.)
        self.solver_params = None
        self.path_out = " "
        self. thetab = None
        self.rhob = None
        self.Pib = None
        self.theta_init_pert = 0
        self.sponge_fct = False
        self.Parameters = Parameters()
        self.slice_3D = slice_3D
        self.exact_sol = None
        self.n = FacetNormal(mesh)
        self.mesh_periodic = mesh_periodic
        self.checkpointing = False
        self.checkpoint_path= " "
        self.check_freq = 50 #checkpointing frequency, checkpoint every 50th time step
        self.out_error = False
        self.error_file = " "

        self.rhs = rhs
        if self.rhs:
            print("rhs defined")
            self.rhs_u = None
            self.rhs_rho = None
            self.rhs_theta = None
        
        self.dim = self.mesh.topological_dimension()
        zvec = [0.0] * self.dim
        zvec[self.dim - 1] = 1.0
        self.zvec = Constant(zvec)

        if self.slice_3D:
            self.V0, self.Vv, self.Vp, self.Vt, self.Vtr = build_spaces_slice_3D(self.mesh, self.vertical_degree, self.horizontal_degree)
        else:
            self.V0, self.Vv, self.Vp, self.Vt, self.Vtr = build_spaces(self.mesh, self.vertical_degree, self.horizontal_degree)

        self.W = self.V0*self.Vp*self.Vt*self.Vtr

        self.Pi0 = Function(self.Vp)
        self.rho0 = Function(self.Vp)
        self.lambdar0 = Function(self.Vtr)
        self.cp = self.Parameters.cp
        self.N = self.Parameters.N
        self.g = self.Parameters.g

    def solve(self, dt, tmax, dumpt, hydrostatic_balance_background = True):

        self.theta_b = Function(self.Vt).interpolate(self.thetab)
        
        if hydrostatic_balance_background:
            compressible_hydrostatic_balance_with_correct_pi_top(
                        self.mesh,
                        self.vertical_degree, self.horizontal_degree,
                        self.Parameters,
                        self.theta_b, self.rho0, self.lambdar0, self.Pi0, 
                        self.slice_3D)
        else: 
            self.rho0 = Function(self.Vp, name="rho0").interpolate(self.rhob)
            self.Pi0 = Function(self.Vp, name="Pi0").interpolate(self.Pib)
            self.lambdar0 = Function(self.Vtr, name = "lambdar0").project(self.Pib)

        Un = Function(self.W, name = "Un")
        Unp1 = Function(self.W, name = "Unp1")
        un, rhon, thetan, lambdarn = (Un).split()
        unp1, rhonp1, thetanp1, lambdarnp1 = split(Unp1)

        unph = 0.5*(un + unp1)
        thetanph = 0.5*(thetan + thetanp1)
        rhonph = 0.5*(rhon + rhonp1)
        
        Pin = thermodynamics_pi(rhon, thetan)
        Pinp1 = thermodynamics_pi(rhonp1, thetanp1)
        Pinph = 0.5*(Pin + Pinp1)

        # functions for the upwinding terms
        unph_mean = 0.5*(unph("+") + unph("-"))

        dS = dS_h + dS_v

        def unn(r):
            return 0.5*(
                        dot(unph_mean, self.n(r))
                        + abs(dot(unph_mean, self.n(r)))
                        )

        def Upwind(r):
            return 0.5*(sign(dot(unph_mean, self.n(r)))+1)

        def unn_vert(r):
            return 0.5*(
                        dot(unph(r), self.n(r))
                        + abs(dot(unph(r), self.n(r)))
                        )

        def Upwind_vert(r):
            return 0.5*(sign(dot(unph(r), self.n(r)))+1)


        if self.dim == 2:
            perp_u_upwind_vert = lambda q: Upwind_vert('+')*perp(q('+')) + Upwind_vert('-')*perp(q('-'))
            perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))

            def uadv_eq_vec_inv(w):
                return(-inner(perp(grad(inner(w, perp(unph)))), unph)*dx
                       - inner(jump(inner(w, perp(unph)), self.n), perp_u_upwind(unph))*(dS_v)
                       - inner(jump(inner(w, perp(unph)), self.n), perp_u_upwind(unph))*(dS_h)
                       # - inner(inner(w, perp(unph))* self.n, unph) * ( ds_t + ds_b )
                       - 0.5 * inner(unph, unph) * div(w) * dx
                       # + 0.5 * inner(u_upwind(unph), u_upwind(unph)) * jump(w, n) * dS_h
                       # +Constant(1e-50)*jump(unph,self.n)*jump(w,self.n)*dS_h
                       # +Constant(1e-50)*inner(unph,self.n)*inner(w,self.n)*ds_tb
                       )
            def uadv_eq(w):
                return (-inner(div(outer(w, unph)), unph)*dx 
                        + dot(jump(w),(unn('+')*unph('+') - unn('-')*unph('-')))*dS)
        
        elif self.dim == 3:

            def uadv_eq(w):
                return(inner(unph, curl(cross(unph, w)))*dx
                       - inner(Upwind('+')*unph('+') + Upwind('-')*unph('-'), 
                       (cross(self.n, cross(unph, w)))('+') + (cross(self.n, cross(unph, w)))('-'))*dS
                       - 0.5 * inner(unph, unph) * div(w)*dx
                       )

        else:
            raise NotImplementedError

        def u_eqn(w, gammar):
            if self.g >0:
                gravity_term =  self.g * inner(w, self.zvec)*dx
            else:
                gravity_term = 0
            return (inner(w, unp1 - un)*dx + self.dT * (
                    uadv_eq(w)
                    - self.cp*div(w*thetanph)*Pinph*dx
                    + self.cp*jump(thetanph*w, self.n)*lambdarnp1('+')*dS_h
                    + self.cp*inner(thetanph*w, self.n)*lambdarnp1*(ds_t + ds_b)
                    + self.cp*jump(thetanph*w, self.n)*(0.5*(Pinph('+') + Pinph('-')))*dS_v
                    # + c_p * inner(thetanph * w, n) * Pinph * (ds_v)
                    + gravity_term)
                    + gammar('+')*jump(unph, self.n)*dS_h  # maybe try adding theta
                    + gammar*inner(unph, self.n)*(ds_t + ds_b)
                    )


        dS = dS_h + dS_v

        def rho_eqn(phi):
            return (phi*(rhonp1 - rhon)*dx
                    + self.dT*(-inner(grad(phi), rhonph*unph)*dx
                               + (phi('+') - phi('-'))*(unn('+')*rhonph('+') - unn('-')*rhonph('-'))*dS
                               # + dot(phi*unph,n) *ds_v
                               )
                    )

        def theta_eqn(chi):
            return (chi*(thetanp1 - thetan)*dx
                    + self.dT * (
                            inner(chi*unph, grad(thetanph))*dx
                            + (chi('+') - chi('-'))* (unn('+')*thetanph('+') - unn('-')*thetanph('-'))*dS
                            - dot(chi('+')*thetanph('+')*unph('+'),  self.n('+'))*dS 
                            - inner(chi('-')*thetanph('-')*unph('-'), self.n('-'))*dS
                            # + dot(unph*chi,n)*thetanph * (ds_v + ds_t + ds_b)
                            # - inner(chi*thetanph * unph, n)* (ds_v +  ds_t + ds_b)
                            )
                    )

        # set up test functions and the nonlinear problem
        w, phi, chi, gammar = TestFunctions(self.W)
        # a = Constant(10000.0)
        eqn = u_eqn(w, gammar) + theta_eqn(chi) + rho_eqn(phi) # + gamma * rho_eqn(div(w))

        if self.rhs:
            eqn += inner(self.rhs_u, w)*dx
            eqn += self.rhs_rho*phi*dx
            eqn += self.rhs_theta*chi*dx
            print("RHS added to equatioin")

        if self.dim == 3:
            f = Constant(1.0e-4)
            eqn += f*inner(w, cross(self.zvec, unph))*dx

        if self.dim == 2:
            x, z = SpatialCoordinate(self.mesh)
        elif self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            raise NotImplementedError

        if self.sponge_fct:
            mubar = 0.15
            zc = self.H-10000.
            
            mu_top = conditional(z <= zc, 0.0, mubar*sin((np.pi/2.)*(z-zc)/(self.H-zc))**2)
            mu = Function(self.Vp).interpolate(mu_top/self.dT)
            eqn += self.dT*mu*inner(w, self.zvec)*inner(unph, self.zvec)*dx

        nprob = NonlinearVariationalProblem(eqn, Unp1)

        if not self.mesh_periodic:
            bc1 = DirichletBC(self.W.sub(0), 0., 1)
            bc2 = DirichletBC(self.W.sub(0), 0., 2)
            nprob = NonlinearVariationalProblem(eqn, Unp1, bcs=[bc1, bc2])
        self.solver = NonlinearVariationalSolver(nprob, solver_parameters=self.solver_params)

        theta0 = Function(self.Vt, name="theta0").interpolate(self.thetab + self.theta_init_pert)
        self.thetab = Function(self.Vt, name = "thetab").interpolate(self.thetab)
        self.rho0 = Function(self.Vp, name="rhob").interpolate(self.rho0)  # where rho_b solves the hydrostatic balance eq.

        Pi_out = Function(self.Vp).interpolate(thermodynamics_pi(self.rho0, self.thetab))
        file = File("ppi0.pvd")
        #Pi0_out = Function(self.Vp).interpolate(Pin)
        file.write(Pi_out)
        
        
        #U0_bc = apply_BC_def_mesh(self.u0, self.V0, self.Vtr)
        #u0_bc, _ = U0_bc.split()
        #u0 = Function(self.V0, name="u0").project(u0_bc)
        u0 = Function(self.V0, name="u0").project(self.u0)

        un.assign(u0)
        rhon.assign(self.rho0)
        thetan.assign(theta0)
        lambdarn.assign(self.lambdar0)

        PETSc.Sys.Print("rho max min", rhon.dat.data.max(),  rhon.dat.data.min())
        PETSc.Sys.Print("theta max min", thetan.dat.data.max(), thetan.dat.data.min())
        PETSc.Sys.Print("lambda max min", lambdarn.dat.data.max(), lambdarn.dat.data.min())

        # initial guess for Unp1 is Un
        Unp1.assign(Un)

        #COURANT NUmber
        def both(u):
            return 2*avg(u)

        DG0 = FunctionSpace(self.mesh, "DG", 0)
        One = Function(DG0).assign(1.0)
        v = TestFunction(DG0)
        Courant_num = Function(DG0, name="Courant numerator")
        
        Courant_num_form = self.dT*(
            (unn('+')*v('+') + unn('-')*v('-'))*(dS_v + dS_h)
            #+ (unn('+')*v('+'))*ds_tb
        )
        Courant_denom = Function(DG0, name="Courant denominator")
        assemble(One*v*dx, tensor=Courant_denom)
        Courant = Function(DG0, name="Courant")

        assemble(Courant_num_form, tensor=Courant_num)
        courant_frac = Function(DG0).interpolate(Courant_num/Courant_denom)
        Courant.assign(courant_frac)
        #Courant.assign(Courant_num/Courant_denom)


        file = File(self.path_out + "_backgroundfcts.pvd")
        file.write(self.rho0, self.thetab, u0, Courant)
    
        if self.checkpointing:
            with CheckpointFile("backgroundfcts_new.h5", 'w') as fileb:
                fileb.save_mesh(self.mesh)
                fileb.save_function(self.thetab)
                fileb.save_function(self.rho0)
        print("BACKGROUND SAVED")
        

        file_out = File(self.path_out + '.pvd')

        rhon_pert = Function(self.Vp)
        thetan_pert = Function(self.Vt)
        rhon_pert.assign(rhon - self.rho0)
        thetan_pert.assign(thetan - self.theta_b)

        file_out.write(un, rhon_pert, thetan_pert, Courant)

        file = File(self.path_out + '_nonpert.pvd')
        file.write(un, rhon, thetan, Courant)

        tdump = 0.
        self.dT.assign(dt)
       

        PETSc.Sys.Print('tmax', tmax, 'dt', dt)
        t = 0.
        step = 0
        idx_count =0

        
        if self.checkpointing:
            print("CHECKPOINTING")
            with CheckpointFile(self.checkpoint_path, 'w') as afile:
                afile.save_mesh(self.mesh)
                afile.save_function(Un, idx = idx_count)
            idx_count += 1 
            if self.exact_sol:
                u_exact, rho_exact = self.exact_sol(x,z,t)
                uerror = norm(un-u_exact)/norm(u_exact)
                rhoerror = norm(rhon-rho_exact)/norm(rho_exact)
                with open(self.path_out+"_uerrors.txt", 'w') as file_uerr:
                    file_uerr.write(str(uerror)+'\n')
                with open(self.path_out+"_rhoerrors.txt", 'w') as file_rhoerr:
                    file_rhoerr.write(str(rhoerror)+'\n')

        while t < tmax - 0.5*dt:
            step+=1
            t += dt
            PETSc.Sys.Print(t)
            tdump += dt

            self.solver.solve()

            Un.assign(Unp1)
            
            if self.checkpointing:
                if step%self.check_freq==0:
                    print("CHECKPOINTING for time t ", t, "with index", idx_count)
                    with CheckpointFile(self.checkpoint_path, 'a') as afile:
                            afile.save_function(Un, idx = idx_count)
                    idx_count += 1 
                    if self.exact_sol:
                        u_exact, rho_exact = self.exact_sol(x,z,t)
                        uerror = norm(un-u_exact)/norm(u_exact)
                        rhoerror = norm(rhon-rho_exact)/norm(rho_exact)
                        #rhoerror = norm(rhon - self.rho0)/norm(self.rho0)
            
                        with open(self.path_out+"_uerrors.txt", 'a') as file_uerr:
                            file_uerr.write(str(uerror)+'\n')
                        with open(self.path_out+"_rhoerrors.txt", 'a') as file_rhoerr:
                            file_rhoerr.write(str(rhoerror)+'\n')


            rhon_pert.assign(rhon - self.rho0)
            thetan_pert.assign(thetan - self.theta_b)

            assemble(Courant_num_form, tensor=Courant_num)
            courant_frac = Function(DG0).interpolate(Courant_num/Courant_denom)
            Courant.assign(courant_frac)

            PETSc.Sys.Print("rho max min pert", rhon_pert.dat.data.max(),  rhon_pert.dat.data.min())
            PETSc.Sys.Print("theta max min pert", thetan_pert.dat.data.max(), thetan_pert.dat.data.min())
            PETSc.Sys.Print('lambda max min', lambdarn.dat.data.max(), lambdarn.dat.data.min())

            if tdump > dumpt - dt*0.5:
                file_out.write(un, rhon_pert, thetan_pert, Courant)
                tdump -= dumpt
            PETSc.Sys.Print(self.solver.snes.getIterationNumber())
            #step+=1
                    
            
        print("Total index count = ", idx_count)
        print("number time steps", step)