from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mesh = PeriodicUnitSquareMesh(40, 40)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "DG", 1)
W = FunctionSpace(mesh, "BDM", 2)

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.2, 0))
ubar = Function(W).interpolate(velocity)

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

D = Function(V).interpolate(1.0 + bell)
D_init = Function(V).assign(D)

# Next we'll create a list to store the function values at every timestep so that
# we can make a movie of them later. ::

Ds = []

# We will run for time :math:`2\pi`, a full rotation.  We take 600 steps, giving
# a timestep close to the CFL limit.  We declare an extra variable ``dtc``; for
# technical reasons, this means that Firedrake does not have to compile new C code
# if the user tries different timesteps.  Finally, we define the inflow boundary
# condition, :math:`q_\mathrm{in}`.  In general, this would be a ``Function``, but
# here we just use a ``Constant`` value. ::

T = 2*math.pi
dt = T/600.0
dtc = Constant(dt)
D_in = Constant(1.0)

# Now we declare our variational forms.  Solving for :math:`\Delta q` at each
# stage, the explicit timestepping scheme means that the left hand side is just a
# mass matrix. ::

dD_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dD_trial*dx

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

L1 = dtc*(eq_D(ubar))
# In our Runge-Kutta scheme, the first step uses :math:`q^n` to obtain
# :math:`q^{(1)}`.  We therefore declare similar forms that use :math:`q^{(1)}`
# to obtain :math:`q^{(2)}`, and :math:`q^{(2)}` to obtain :math:`q^{n+1}`. We
# make use of UFL's ``replace`` feature to avoid writing out the form repeatedly. ::

D1 = Function(V); D2 = Function(V)
L2 = replace(L1, {D: D1}); L3 = replace(L1, {D: D2})

# We now declare a variable to hold the temporary increments at each stage. ::

dD = Function(V)


params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dD)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dD)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dD)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

# We now run the time loop.  This consists of three Runge-Kutta stages, and every
# 20 steps we write out the solution to file and print the current time to the
# terminal. ::

t = 0.0
step = 0
output_freq = 20
while t < T - 0.5*dt:
    solv1.solve()
    D1.assign(D + dD)

    solv2.solve()
    D2.assign(0.75*D + 0.25*(D1 + dD))

    solv3.solve()
    D.assign((1.0/3.0)*D + (2.0/3.0)*(D2 + dD))

    step += 1
    t += dt

    if step % output_freq == 0:
        Ds.append(D.copy(deepcopy=True))
        print("t=", t)

# To check our solution, we display the normalised :math:`L^2` error, by comparing
# to the initial condition. ::

L2_err = sqrt(assemble((D - D_init)*(D - D_init)*dx))
L2_init = sqrt(assemble(D_init*D_init*dx))
print(L2_err/L2_init)

# Finally, we'll animate our solution using matplotlib. We'll need to evaluate
# the solution at many points in every frame of the animation, so we'll employ a
# helper class that pre-computres some relevant data in order to speed up the
# evaluation. ::

nsp = 16
fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)

# We first set up a figure and axes and draw the first frame. ::

fig, axes = plt.subplots()
axes.set_aspect('equal')
colors = tripcolor(D_init, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
fig.colorbar(colors)

# Now we'll create a function to call in each frame. This function will use the
# helper object we created before. ::

def animate(q):
    colors.set_array(fn_plotter(q))

# The last step is to make the animation and save it to a file. ::

interval = 1e3 * output_freq * dt
animation = FuncAnimation(fig, animate, frames=Ds, interval=interval)
try:
    animation.save("DG_advection_test.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")
