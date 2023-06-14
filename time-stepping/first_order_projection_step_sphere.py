from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np



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
ubar = Function(V1).interpolate(u_expr)
file.write(ubar)


# define velocity field to be advected:
x_c = as_vector([1., 0., 0.])
F_0 = Constant(3.)
l_0 = Constant(0.25)

def dist_sphere(x, x_c):
    return acos(dot(x/R0,x_c))


F_theta = F_0*exp(-dist_sphere(x,x_c)**2/l_0**2)
D_expr = conditional(dist_sphere(x,x_c) > 0.5, 0., F_theta)

#D_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
Dn = Function(V2, name = "D").interpolate(D_expr)
Dbar = Function(V2).assign(Dn)
#un starts as ubar, not this example

#un = Function(V1_broken, name = "u").interpolate(velocity)
un = Function(V1, name = "u").assign(ubar)
print("u is set", norm(un), norm(Dn))

Unp1 = Function(W)
unp1, Dnp1 = Unp1.subfunctions

Dnp1.assign(Dn)
unp1.assign(un)




#to be the solutions, initialised with un, Dn
D = Function(V2).assign(Dn)
u = Function(V1_broken).project(un)



T = 86400.
dt = 18.
dtc = Constant(dt)



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

outward_normals = CellNormal(mesh)
def perp(u):
    return cross(outward_normals, u)

#equations for the advection step
def eq_D(ubar):
    uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
    return (inner(grad(phi), ubar)*D*dx
            - jump(phi)*(uup('+')*D('+')
                            - uup('-')*D('-'))*dS)

def adv_u(ubar):
    unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

    return(-inner(div(outer(w, ubar)), u)*dx
           +dot(jump(w), (unn('+')*u('+') - unn('-')*u('-')))*dS
    )

L1_D = dtc*(eq_D(ubar))

eq_u = adv_u(ubar)
#adjust to sphere
unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
eq_u += unn('+')*inner(w('-'), n('+')+n('-'))*inner(u('+'), n('+'))*dS
eq_u += unn('-')*inner(w('+'), n('+')+n('-'))*inner(u('-'), n('-'))*dS

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
Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*x[2]/Constant(R0)  # Coriolis parameter

v, rho = TestFunctions(W)
def proj_u():
    return (-div(v)*g*Dnp1*dx
            + inner(v, f*perp(unp1))*dx 
    )

def proj_D(Dbar):
    uup = 0.5 * (dot(unp1, n) + abs(dot(unp1, n)))
    return (inner(grad(rho), unp1*Dbar)*dx
            - jump(rho)*(uup('+')*Dbar('+')
                            - uup('-')*Dbar('-'))*dS)

#make signs consistent

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
out_file = File("Results/proj_solution.pvd")
out_file.write(Dn, un)


unp1, Dnp1 = Unp1.subfunctions
while t < T - 0.5*dt:

    #u.project(un)
    #D.assign(Dn)

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

    Dn.assign(Dnp1)
    un.assign(unp1)
    Dbar.assign(Dn)
    ubar.assign(un)

    step += 1
    t += dt

    if step % output_freq == 0:
        out_file.write(Dn, un)
        print("t=", t)





