from firedrake import *
from tools.hydrostatic_balance import *
from tools. new_spaces import *
from tools.physics import *
import numpy as np

class compressibleEulerEquations:

    def __init__(self, mesh, vertical_degree=1, horizontal_degree=1):
    

        self.mesh = mesh
        self.vertical_degree = vertical_degree
        self.horizontal_degree = horizontal_degree
        self.H = None
        self.u0 = None
        self. dT = Constant(5.)
        self.solver_params = None
        self.path_out = " "
        self. thetab = None
        self.theta_init_pert = 0
        self.sponge_fct = False
        self.Parameters = Parameters()

        self.n = FacetNormal(mesh)
        
        self.zvec = as_vector([0, 1])

        self.V0, self.Vv, self.Vp, self.Vt, self.Vtr = build_spaces(self.mesh, self.vertical_degree, self.horizontal_degree)

        #self.V0, self.Vv, self.Vp, self.Vt, self.Vtr = build_spaces(self.mesh, self.vertical_degree, self.horizontal_degree),
        self.W = self.V0*self.Vp*self.Vt*self.Vtr

        
        self.Pi0 = Function(self.Vp)
        self.rho0 = Function(self.Vp)
        self.lambdar0 = Function(self.Vtr)
        self.cp = self.Parameters.cp
        self.N = self.Parameters.N
        self.g = self.Parameters.g

    
        
    #initialise rho0 and lambda0
    
    def solve(self, dt, tmax, dumpt):
        
        self.theta_b = Function(self.Vt).interpolate(self.thetab)
        compressible_hydrostatic_balance_with_correct_pi_top(self.mesh, 
                                                                self.vertical_degree, self.horizontal_degree,
                                                                self.Parameters, 
                                                                self.theta_b, self.rho0, self.lambdar0, self.Pi0)
        
        Un = Function(self.W)
        Unp1 = Function(self.W)
        un, rhon, thetan, lambdarn = (Un).split()
        unp1, rhonp1, thetanp1, lambdarnp1 = split(Unp1)

        unph = 0.5*(un + unp1)
        thetanph = 0.5*(thetan + thetanp1)
        lambdarnph = 0.5*(lambdarn + lambdarnp1)
        rhonph = 0.5*(rhon + rhonp1)

        Pin = thermodynamics_pi(rhon, thetan)
        Pinp1 = thermodynamics_pi(rhonp1, thetanp1)
        Pinph = 0.5*(Pin + Pinp1)

        #functions for the upwinding terms
        unn = 0.5*(dot(unph, self.n) + abs(dot(unph, self.n)))  # used for the upwind-term
        Upwind = 0.5*(sign(dot(unph, self.n))+1)
        perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
        u_upwind = lambda q: Upwind('+')*q('+') + Upwind('-')*q('-')

        def uadv_eq(w):
            return(-inner(perp(grad(inner(w, perp(unph)))), unph)*dx
                - inner(jump(inner(w, perp(unph)), self.n), perp_u_upwind(unph))*(dS)
                # - inner(inner(w, perp(unph))* n, unph) * ( ds_t + ds_b )
                - 0.5 * inner(unph, unph) * div(w) * dx
                # + 0.5 * inner(u_upwind(unph), u_upwind(unph)) * jump(w, n) * dS_h
                )

        def u_eqn(w, gammar):
            return (inner(w, unp1 - un)*dx + self.dT * (
                    uadv_eq(w)
                    - self.cp*div(w*thetanph)*Pinph*dx
                    + self.cp*jump(thetanph*w, self.n)*lambdarnp1('+')*dS_h
                    + self.cp*inner(thetanph*w, self.n)*lambdarnp1*(ds_t + ds_b)
                    + self.cp*jump(thetanph*w, self.n)*(0.5*(Pinph('+') + Pinph('-')))*dS_v
                    # + c_p * inner(thetanph * w, n) * Pinph * (ds_v)
                    + gammar('+')*jump(unph, self.n)*dS_h
                    + gammar*inner(unph, self.n)*(ds_t + ds_b)
                    + self.g * inner(w, self.zvec)*dx)
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
                    + self.dT * (inner(chi*unph, grad(thetanph))*dx
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

        if self.sponge_fct:
            mubar = 0.3
            zc = self.H-10000.
            _, z = SpatialCoordinate(self.mesh)
            mu_top = conditional(z <= zc, 0.0, mubar*sin((np.pi/2.)*(z-zc)/(self.H-zc))**2)
            mu = Function(self.Vp).interpolate(mu_top/self.dT)
            eqn += mu*inner(w, self.zvec)*inner(unph, self.zvec)*dx

        nprob = NonlinearVariationalProblem(eqn, Unp1)

        self.solver = NonlinearVariationalSolver(nprob, solver_parameters=self.solver_params)

        theta0 = Function(self.Vt, name="theta0").interpolate(self.thetab + self.theta_init_pert)
        self.thetab = Function(self.Vt).interpolate(self.thetab)
        self.rho0 = Function(self.Vp, name="rho0").interpolate(self.rho0)  # where rho_b solves the hydrostatic balance eq.
        u0 = Function(self.V0, name="u0").project(self.u0)
        #U0_bc = apply_BC_def_mesh(self.u0, self.V0, self.Vtr)
        #u0_bc, _ = U0_bc.split()
        #u0 = Function(self.V0, name="u0").project(u0_bc)

        #self.lambdar0 = Function(self.Vtr, name="lambdar0").assign(self.lambdar0)

        un.assign(u0)
        rhon.assign(self.rho0)
        thetan.assign(theta0)
        lambdarn.assign(self.lambdar0)

        print("rho max min", rhon.dat.data.max(),  rhon.dat.data.min())
        print("theta max min", thetan.dat.data.max(), thetan.dat.data.min())
        print("lambda max min", lambdarn.dat.data.max(), lambdarn.dat.data.min())

        # initial guess for Unp1 is Un
        Unp1.assign(Un)

        file_out = File(self.path_out + '.pvd')

        rhon_pert = Function(self.Vp)
        thetan_pert = Function(self.Vt)
        rhon_pert.assign(rhon - self.rho0)
        thetan_pert.assign(thetan - self.theta_b)

        file_out.write(un, rhon_pert, thetan_pert)

       
        tdump = 0.
        self.dT.assign(dt)
        

        print('tmax', tmax, 'dt', dt)
        t = 0.


        while t < tmax - 0.5*dt:
            print(t)
            t += dt
            tdump += dt
            
            self.solver.solve()

            Un.assign(Unp1)

            rhon_pert.assign(rhon - self.rho0)
            thetan_pert.assign(thetan - self.theta_b)

            print("rho max min pert", rhon_pert.dat.data.max(),  rhon_pert.dat.data.min())
            print("theta max min pert", thetan_pert.dat.data.max(), thetan_pert.dat.data.min())
            print('lambda max min', lambdarn.dat.data.max(), lambdarn.dat.data.min())
            
            if tdump > dumpt - dt*0.5:
                file_out.write(un, rhon_pert, thetan_pert)
                # file_gw.write(un, rhon, thetan, lambdarn)
                # file2.write(un_pert, rhon_pert, thetan_pert)
                tdump -= dumpt
 
'''
class CompressibleEulerSolver(compressibleEulerSetUp):

    def __init__(self, mesh, u0, theta_b, theta_initial_pert=0, vertical_degree=1, horizontal_degree=1, dt=5., sponge_fct=False):

        compressibleEulerSetUp.__init__(self, mesh, u0, theta_b, vertical_degree, horizontal_degree, dt)
       
        un, rhon, thetan, lambdarn = self.Un.split()
        unp1, rhonp1, thetanp1, lambdarnp1 = self.Unp1.split()

        unph = 0.5*(un + unp1)
        thetanph = 0.5*(thetan + thetanp1)
        lambdarnph = 0.5*(lambdarn + lambdarnp1)
        rhonph = 0.5*(rhon + rhonp1)

        Pin = thermodynamics_pi(parameters, rhon, thetan)
        Pinp1 = thermodynamics_pi(parameters, rhonp1, thetanp1)
        Pinph = 0.5*(Pin + Pinp1)

        #functions for the upwinding terms
        unn = 0.5*(dot(unph, self.n) + abs(dot(unph, self.n)))  # used for the upwind-term
        Upwind = 0.5*(sign(dot(unph, self.n))+1)
        perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
        u_upwind = lambda q: Upwind('+')*q('+') + Upwind('-')*q('-')

        def uadv_eq(w):
            return(-inner(perp(grad(inner(w, perp(unph)))), unph)*dx
                - inner(jump(inner(w, perp(unph)), self.n), perp_u_upwind(unph))*(dS)
                # - inner(inner(w, perp(unph))* n, unph) * ( ds_t + ds_b )
                - 0.5 * inner(unph, unph) * div(w) * dx
                # + 0.5 * inner(u_upwind(unph), u_upwind(unph)) * jump(w, n) * dS_h
                )

        def u_eqn(w, gammar):
            return (inner(w, unp1 - un)*dx + self.dT * (
                    uadv_eq(w)
                    - self.cp*div(w*thetanph)*Pinph*dx
                    + self.cp*jump(thetanph*w, self.n)*lambdarnp1('+')*dS_h
                    + self.cp*inner(thetanph*w, self.n)*lambdarnp1*(ds_t + ds_b)
                    + self.cp*jump(thetanph*w, self.n)*(0.5*(Pinph('+') + Pinph('-')))*dS_v
                    # + c_p * inner(thetanph * w, n) * Pinph * (ds_v)
                    + gammar('+')*jump(unph, self.n)*dS_h
                    + gammar*inner(unph, self.n)*(ds_t + ds_b)
                    + self.g * inner(w, self.zvec)*dx)
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
                    + self.dT * (inner(chi*unph, grad(thetanph))*dx
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

        if sponge_fct:
            mubar = 0.3
            zc = self.H-10000.
            _, z = SpatialCoordinate(mesh)
            mu_top = conditional(z <= zc, 0.0, mubar*sin((np.pi/2.)*(z-zc)/(H-zc))**2)
            mu = Function(self.Vp).interpolate(mu_top/self.dT)
            eqn += mu*inner(w, self.zvec)*inner(unph, self.zvec)*dx

        nprob = NonlinearVariationalProblem(eqn, self.Unp1)
        self.solver = NonlinearVariationalSolver(nprob, self.solver_params)

        theta0 = Function(self.Vt, name="theta0").interpolate(theta_b + theta_initial_pert)
        self.rho0 = Function(self.Vp, name="rho0").interpolate(self.rho0)  # where rho_b solves the hydrostatic balance eq.
        u0 = Function(self.V0, name="u0").project(u0)
        #U0_bc = apply_BC_def_mesh(self.u0, self.V0, self.Vtr)
        #u0_bc, _ = U0_bc.split()
        #u0 = Function(self.V0, name="u0").project(u0_bc)

        self.lambdar0 = Function(self.Vtr, name="lambdar0").assign(self.lambdar0)

        un.assign(u0)
        rhon.assign(self.rho0)
        thetan.assign(theta0)
        lambdarn.assign(self.lambdar0)

        print("rho max min", rhon.dat.data.max(),  rhon.dat.data.min())
        print("theta max min", thetan.dat.data.max(), thetan.dat.data.min())
        print("lambda max min", lambdarn.dat.data.max(), lambdarn.dat.data.min())

        # initial guess for Unp1 is Un
        self.Unp1.assign(self.Un)

        file_out = File(self.path_out + '.pvd')

        rhon_pert = Function(self.Vp)
        thetan_pert = Function(self.Vt)
        rhon_pert.assign(rhon - self.rho0)
        thetan_pert.assign(thetan - self.theta0)

        file_out.write(un, rhon_pert, thetan_pert)

        dt = 5.
        dumpt = 100.
        tdump = 0.
        self.dT.assign(dt)
        tmax = 15000.

        print('tmax', tmax, 'dt', dt)
        t = 0.


        while t < tmax - 0.5*dt:
            print(t)
            t += dt
            tdump += dt
            
            self.solver.solve()

            self.Un.assign(self.Unp1)

            rhon_pert.assign(rhon - self.rho0)
            thetan_pert.assign(thetan - theta0)

            print("rho max min pert", rhon_pert.dat.data.max(),  rhon_pert.dat.data.min())
            print("theta max min pert", thetan_pert.dat.data.max(), thetan_pert.dat.data.min())
            
            if tdump > dumpt - dt*0.5:
                file_out.write(un, rhon_pert, thetan_pert)
                # file_gw.write(un, rhon, thetan, lambdarn)
                # file2.write(un_pert, rhon_pert, thetan_pert)
                tdump -= dumpt


def return_solver(self):
    return self.solver
'''