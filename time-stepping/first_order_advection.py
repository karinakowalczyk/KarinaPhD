from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
   
def plot_fd_vector(function, name):
    fig, axes = plt.subplots()
    quiver(function, axes=axes)
    axes.set_aspect("equal")
    axes.set_title(name)
    fig.savefig(name)

def plot_fd_scalar(function, file_name):
    # We first set up a figure and axes and draw the first frame. ::
    nsp=16
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(function, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
    fig.colorbar(colors)
    fig.savefig(file_name)


mesh = PeriodicUnitSquareMesh(40,40)

n = FacetNormal(mesh)

#velocity space
element = FiniteElement("BDM", triangle, degree=2)
V1_broken = FunctionSpace(mesh, BrokenElement(element))
V1 = FunctionSpace(mesh, element)
#space for height:
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1_broken,V2))

#set up test and trial functions
u = Function(V1_broken)
D= Function(V2)

#initialise u, D, will hold un, Dn later
x, y = SpatialCoordinate(mesh)
#velocity = as_vector((0.5 - y, x - 0.5))
velocity = as_vector((0.3,0))
ubar = Function(V1).interpolate(velocity)

plot_fd_vector(ubar, 'ubar')

#cosine-bell as initial condition for D
bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
bell = 0.4*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
D.interpolate(1+bell)

plot_fd_scalar(D, "D_init")

D_init = Function(V2).assign(D)
Dbar = Function(V2).assign(0.)


#D.interpolate(Constant(0.))
Ds = []
us = []


T = 1.
dt = T/100.0
dtc = Constant(dt)
#q_in = Constant(1.0)

w = TestFunction(V1_broken)
phi = TestFunction(V2)

#set up equations
def perp_u_upwind(u):
    Upwind = 0.5 * (sign(dot(u, n)) + 1)
    return Upwind('+')*perp(u('+')) + Upwind('-')*perp(u('-'))

def adv_u(ubar):
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

    return(-inner(div(outer(w, ubar)), u)*dx
           +dot(jump(w), (un('+')*u('+') - un('-')*u('-')))*dS
    )

def adv_eq_vector_id(u, ubar, w):
    return (
        + -inner(perp(grad(inner(w, perp(ubar)))), u)*dx
        - inner(jump(inner(w, perp(ubar)), n), perp_u_upwind(u))*(dS)
        - inner(ubar, u) * div(w) * dx
        )

def eq_D(Dbar, ubar):
    h = D-Dbar
    uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
    return (- inner(grad(phi), outer(D,ubar))*dx
            + dot(jump(phi),(uup('+')*D('+')
                            - uup('-')*D('-')))*(dS)
    )

#equations are independent
L1_u = dtc*adv_u(ubar)
L1_D = dtc*eq_D(Dbar,ubar)

#for the second and third step of SSPRK
u1 = Function(V1_broken); u2 = Function(V1_broken)
L2_u = replace(L1_u, {u: u1}); L3_u = replace(L1_u, {u: u2})

D1 = Function(V2); D2 = Function(V2)
L2_D = replace(L1_D, {D: D1}); L3_D = replace(L1_D, {D: D2})

#time derivative term involves just a mass matrix
du = TrialFunction(V1_broken)
a_u = inner(du,w)*dx

dD = TrialFunction(V2)
a_D = dD*phi*dx

#function to hold current increment of u
du = Function(V1_broken)
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob_u1 = LinearVariationalProblem(a_u, L1_u, du)
solv_u1 = LinearVariationalSolver(prob_u1, solver_parameters=params)
prob_u2 = LinearVariationalProblem(a_u, L2_u, du)
solv_u2 = LinearVariationalSolver(prob_u2, solver_parameters=params)
prob_u3 = LinearVariationalProblem(a_u, L3_u, du)
solv_u3 = LinearVariationalSolver(prob_u3, solver_parameters=params)

#function to hold current increment of D
dD = Function(V2)
solv_D1 = LinearVariationalSolver(prob_u1, solver_parameters=params)
prob_D1 = LinearVariationalProblem(a_D, L1_D, dD)
prob_D2 = LinearVariationalProblem(a_D, L2_D, dD)
solv_D2 = LinearVariationalSolver(prob_u2, solver_parameters=params)
prob_D3 = LinearVariationalProblem(a_D, L3_D, dD)
solv_D3 = LinearVariationalSolver(prob_u3, solver_parameters=params)


t = 0.0
step = 0
output_freq = 10
while t < T - 0.5*dt:
    #solving the u equation
    solv_u1.solve()
    u1.assign(u + du)

    solv_u2.solve()
    u2.assign(0.75*u + 0.25*(u1 + du))

    solv_u3.solve()
    u.assign((1.0/3.0)*u + (2.0/3.0)*(u2 + du))

    #solving the D-equation
    solv_D1.solve()
    D1.assign(D + dD)

    solv_D2.solve()
    D2.assign(0.75*D + 0.25*(D1 + dD))

    solv_D3.solve()
    D.assign((1.0/3.0)*D + (2.0/3.0)*(D2 + dD))

    

    step += 1
    t += dt
    if step % output_freq == 0:
        Ds.append(D.copy(deepcopy=True))
        us.append(u.copy(deepcopy=True))
        print("t=", t)


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
    animation.save("DG_advection.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")
