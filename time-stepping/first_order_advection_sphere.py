from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
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


#velocity space
element = FiniteElement("BDM", triangle, degree=2)
V1_broken = FunctionSpace(mesh, BrokenElement(element))
V1 = FunctionSpace(mesh, element)
#space for height:
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1_broken,V2))

# SET UP EXAMPLE


u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
#solid body rotation (?)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
file = File('ubar.pvd')
ubar = Function(V1).interpolate(u_expr)
file.write(ubar)

D_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
D = Function(V2, name = "D").interpolate(D_expr)


# define velocity field to be advected:
x_c = as_vector([1., 0., 0.])
F_0 = Constant(3.)
l_0 = Constant(0.25)

def dist_sphere(x, x_c):
    return acos(dot(x/R0,x_c))

e_theta = as_vector([x[0]*x[2], x[1]*x[2], -x[0]**2 - x[1]**2])
e_theta = e_theta/(norm(e_theta)+1e-8)
print("etheta norm:", norm(e_theta))

F_theta = F_0*exp(-dist_sphere(x,x_c)**2/l_0**2)
F_theta_c = conditional(dist_sphere(x,x_c) > 0.5, 0., F_theta)
velocity = F_theta_c*e_theta

u = Function(V1_broken, name = "u").interpolate(velocity)
print("u is set", norm(u), norm(D))

D_init = Function(V2).assign(D)
u_init = Function(V1_broken).assign(u)
file = File('init.pvd')
file.write(D_init, u_init)

T = 2*86400.
dt = 360.
dtc = Constant(dt)

def both(u):
    return 2*avg(u)

# compute COURANT number
DG0 = FunctionSpace(mesh, "DG", 0)
One = Function(DG0).assign(1.0)
unn = 0.5*(inner(-u, n) + abs(inner(-u, n))) # gives fluxes *into* cell only
v = TestFunction(DG0)
Courant_num = Function(DG0, name="Courant numerator")
Courant_num_form = dt*(both(unn*v)*dS)

Courant_denom = Function(DG0, name="Courant denominator")
assemble(One*v*dx, tensor=Courant_denom)
Courant = Function(DG0, name="Courant")

assemble(Courant_num_form, tensor=Courant_num)
courant_frac = Function(DG0).interpolate(Courant_num/Courant_denom)
Courant.assign(courant_frac)


# We will run for time :math:`2\pi`, a full rotation.  We take 600 steps, giving
# a timestep close to the CFL limit.  We declare an extra variable ``dtc``; for
# technical reasons, this means that Firedrake does not have to compile new C code
# if the user tries different timesteps.  Finally, we define the inflow boundary
# condition, :math:`q_\mathrm{in}`.  In general, this would be a ``Function``, but
# here we just use a ``Constant`` value. ::




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
un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

# We now define our right-hand-side form ``L1`` as :math:`\Delta t` times the
# sum of four integrals.

L1test = dtc*(D*div(phi*ubar)*dx
          #- conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
          #- conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*D('+') - un('-')*D('-'))*dS)
def eq_D(ubar):
    uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
    return (inner(grad(phi), ubar)*D*dx
            - jump(phi)*(uup('+')*D('+')
                            - uup('-')*D('-'))*dS)

def perp_u_upwind(u):
    Upwind = 0.5 * (sign(dot(u, n)) + 1)
    return Upwind('+')*perp(u('+')) + Upwind('-')*perp(u('-'))

un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
def adv_u(ubar):
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

    return(-inner(div(outer(w, ubar)), u)*dx
           +dot(jump(w), (un('+')*u('+') - un('-')*u('-')))*dS
    )

eq_u = adv_u(ubar)
#adjust to sphere

eq_u += un('+')*inner(w('-'), n('+')+n('-'))*inner(u('+'), n('+'))*dS
eq_u += un('-')*inner(w('+'), n('+')+n('-'))*inner(u('-'), n('-'))*dS
L1_u = dtc*(-eq_u)
L1_D = dtc*(eq_D(ubar))
# In our Runge-Kutta scheme, the first step uses :math:`q^n` to obtain
# :math:`q^{(1)}`.  We therefore declare similar forms that use :math:`q^{(1)}`
# to obtain :math:`q^{(2)}`, and :math:`q^{(2)}` to obtain :math:`q^{n+1}`. We
# make use of UFL's ``replace`` feature to avoid writing out the form repeatedly. ::

D1 = Function(V2); D2 = Function(V2)
L2_D = replace(L1_D, {D: D1}); L3_D = replace(L1_D, {D: D2})

u1 = Function(V1_broken); u2 = Function(V1_broken)
L2_u = replace(L1_u, {u: u1}); L3_u = replace(L1_u, {u: u2})


# We now declare a variable to hold the temporary increments at each stage. ::

dD = Function(V2)
du = Function(V1_broken)


params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob_1_D = LinearVariationalProblem(a_D, L1_D, dD)
solv_1_D = LinearVariationalSolver(prob_1_D, solver_parameters=params)
prob_2_D = LinearVariationalProblem(a_D, L2_D, dD)
solv_2_D = LinearVariationalSolver(prob_2_D, solver_parameters=params)
prob_3_D = LinearVariationalProblem(a_D, L3_D, dD)
solv_3_D = LinearVariationalSolver(prob_3_D, solver_parameters=params)

prob_1_u = LinearVariationalProblem(a_u, L1_u, du)
solv_1_u = LinearVariationalSolver(prob_1_u, solver_parameters=params)
prob_2_u = LinearVariationalProblem(a_u, L2_u, du)
solv_2_u = LinearVariationalSolver(prob_2_u, solver_parameters=params)
prob_3_u = LinearVariationalProblem(a_u, L3_u, du)
solv_3_u = LinearVariationalSolver(prob_3_u, solver_parameters=params)

# We now run the time loop.  This consists of three Runge-Kutta stages, and every
# 20 steps we write out the solution to file and print the current time to the
# terminal. ::

t = 0.0
step = 0
output_freq = 2
out_file = File("Results/advection_sphere.pvd")
out_file.write(D, u, Courant)
while t < T - 0.5*dt:
    solv_1_D.solve()
    D1.assign(D + dD)

    solv_2_D.solve()
    D2.assign(0.75*D + 0.25*(D1 + dD))

    solv_3_D.solve()
    D.assign((1.0/3.0)*D + (2.0/3.0)*(D2 + dD))

    solv_1_u.solve()
    u1.assign(u + du)

    solv_2_u.solve()
    u2.assign(0.75*u + 0.25*(u1 + du))

    solv_3_u.solve()
    u.assign((1.0/3.0)*u + (2.0/3.0)*(u2 + du))


    step += 1
    t += dt

    if step % output_freq == 0:
        out_file.write(D,u, Courant)
        print("t=", t)

# To check our solution, we display the normalised :math:`L^2` error, by comparing
# to the initial condition. ::

L2_err = sqrt(assemble((D - D_init)*(D - D_init)*dx))
L2_init = sqrt(assemble(D_init*D_init*dx))
print(L2_err/L2_init)

