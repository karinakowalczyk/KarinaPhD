from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_fd_vector(function, name):
    fig, axes = plt.subplots()
    quiver(function, axes=axes)
    axes.set_aspect("equal")
    axes.set_title(name)
    fig.savefig(name)

R0 = 6371220.
Omega = Constant(7.292e-5)  # rotation rate
#f = 2*Omega*z/Constant(R0)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant

mesh = PeriodicUnitSquareMesh(40, 40)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "DG", 1)
W = FunctionSpace(mesh, "BDM", 2)
#velocity space
element = FiniteElement("BDM", triangle, degree=2)
V1_broken = FunctionSpace(mesh, BrokenElement(element))
V1 = FunctionSpace(mesh, element)
#space for height:
V2 = FunctionSpace(mesh, "DG", 1)
W_broken = MixedFunctionSpace((V1_broken,V2))
W = MixedFunctionSpace((V1,V2))

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)

velocity = as_vector((1., 0))
ubar = Function(V1).interpolate(velocity)

# Now, we set up the cosine-bell--cone--slotted-cylinder initial coniditon. The
# first four lines declare various parameters relating to the positions of these
# objects, while the analytic expressions appear in the last three lines. ::

bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

Unp1 = Function(W)
unp1, Dnp1 = Unp1.subfunctions

Dnp1.interpolate(1.0 + bell)
unp1.interpolate(as_vector([1+bell, 1+bell]))
Dn = Function(V2)
un = Function(V1)
ubar.assign(unp1)
Dbar = Function(V2).assign(Dnp1)

#to be the solutions, initialised with un, Dn
D = Function(V2).assign(Dnp1)
u = Function(V1_broken).project(unp1)
D_init = Function(V2).assign(D)
u_init = Function(V1_broken).assign(u)
plot_fd_vector(u_init, "uinit")
   
# Next we'll create a list to store the function values at every timestep so that
# we can make a movie of them later. ::

Ds = []
us =[]

# We will run for time :math:`2\pi`, a full rotation.  We take 600 steps, giving
# a timestep close to the CFL limit.  We declare an extra variable ``dtc``; for
# technical reasons, this means that Firedrake does not have to compile new C code
# if the user tries different timesteps.  Finally, we define the inflow boundary
# condition, :math:`q_\mathrm{in}`.  In general, this would be a ``Function``, but
# here we just use a ``Constant`` value. ::

T = 1.
dt = T/600.0
dtc = Constant(dt)
D_in = Constant(1.0)

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
unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

# We now define our right-hand-side form ``L1`` as :math:`\Delta t` times the
# sum of four integrals.

L1test = dtc*(D*div(phi*ubar)*dx
          #- conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
          #- conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds
          - (phi('+') - phi('-'))*(unn('+')*D('+') - unn('-')*D('-'))*dS)
def eq_D(ubar):
    uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
    return (inner(grad(phi), ubar)*D*dx
            - jump(phi)*(uup('+')*D('+')
                            - uup('-')*D('-'))*dS)

def perp_u_upwind(u):
    Upwind = 0.5 * (sign(dot(u, n)) + 1)
    return Upwind('+')*perp(u('+')) + Upwind('-')*perp(u('-'))

def adv_u(ubar):
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

    return(-inner(div(outer(w, ubar)), u)*dx
           +dot(jump(w), (un('+')*u('+') - un('-')*u('-')))*dS
    )
L1_D = dtc*(eq_D(ubar))

L1_u = dtc*(-adv_u(ubar))
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

unp1, Dnp1 = split(Unp1)
def proj_u():
    return (-div(v)*g*Dnp1*dx
            #+ inner(v, f*perp(unp1))*dx # doesn't make sense in 2d?
    )

def proj_D(Dbar):
    uup = 0.5 * (dot(unp1, n) + abs(dot(unp1, n)))
    return (inner(grad(rho), unp1)*Dbar*dx
            - jump(rho)*(uup('+')*Dbar('+')
                            - uup('-')*Dbar('-'))*dS)

#make signs consistent
v, rho = TestFunctions(W)
a_proj_u = inner(v, unp1 - u)*dx + dtc* proj_u()

a_proj_D = rho*(Dnp1 -D)*dx - dtc*proj_D(Dbar)

a_proj = a_proj_u + a_proj_D

prob = NonlinearVariationalProblem(a_proj, Unp1)
solver_proj = NonlinearVariationalSolver(prob, solver_parameters=params)


# We now run the time loop.  This consists of three Runge-Kutta stages, and every
# 20 steps we write out the solution to file and print the current time to the
# terminal. ::

t = 0.0
step = 0
output_freq = 20
out_file = File("solution.pvd")
out_file.write(D, u)


unp1, Dnp1 = Unp1.subfunctions
while t < T - 0.5*dt:

    Dn.assign(Dnp1)
    un.assign(unp1)
    Dbar.assign(Dn)
    ubar.assign(un)

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

    # PROJECTION STEP
    Dnp1.assign(D)
    unp1.project(u) #u from discontinuous space

    solver_proj.solve()

    step += 1
    t += dt

    if step % output_freq == 0:
        out_file.write(D,u)
        print("t=", t)



