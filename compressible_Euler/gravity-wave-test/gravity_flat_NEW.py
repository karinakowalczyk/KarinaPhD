from firedrake import *
from tools import *
import sys

'''
    version of "mountain_non-hydrostatic in gusto for new velocity space
    without making use of gusto
'''


# set up mesh and parameters for main computations
dT = Constant(0)  # to be set later
parameters = Parameters()
g = parameters.g
c_p = parameters.cp


# build volume mesh
L = 3.0e5
H = 1.0e4  # Height position of the model top
delx = 1000
delz = 1000
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

H = Constant(H)

# set up fem spaces
V0, _, Vp, Vt, Vtr = build_spaces(mesh, vertical_degree=1, horizontal_degree=1)
W = V0*Vp*Vt*Vtr

# initialise background temperature
# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
N = parameters.N
x, z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic Pi and rho by solving compressible balance equation
# to be used as initial guess for the full solver later
Pi = Function(Vp)
rho_b = Function(Vp)
lambdar_b = Function(Vtr)

compressible_hydrostatic_balance_with_correct_pi_top(mesh, parameters, theta_b, rho_b, lambdar_b, Pi)


# initialise functions for full Euler solver
xc = L/2
a = Constant(5000.)
thetab_pert = 0.01*sin(pi*z/H)/(1+(x-xc)**2/a**2)
theta0 = Function(Vt, name="theta0").interpolate(theta_b + thetab_pert)

rho0 = Function(Vp, name="rho0").interpolate(rho_b)  # where rho_b solves the hydrostatic balance eq.
u0 = Function(V0, name="u0").project(as_vector([20.0, 0.0]))
lambdar0 = Function(Vtr, name="lambdar0").assign(lambdar_b)  # we use lambda from hydrostatic solve as initial guess


# define trial functions
zvec = as_vector([0,1])
n = FacetNormal(mesh)

Un = Function(W)
Unp1 = Function(W)
un, rhon, thetan, lambdarn = Un.split()
unp1, rhonp1, thetanp1, lambdarnp1 = split(Unp1)

unph = 0.5*(un + unp1)
thetanph = 0.5*(thetan + thetanp1)
lambdarnph = 0.5*(lambdarn + lambdarnp1)
rhonph = 0.5*(rhon + rhonp1)

Pin = thermodynamics_pi(parameters, rhon, thetan)
Pinp1 = thermodynamics_pi(parameters, rhonp1, thetanp1)
Pinph = 0.5*(Pin + Pinp1)

# functions for the upwinding terms
unn = 0.5*(dot(unph, n) + abs(dot(unph, n)))  # used for the upwind-term
Upwind = 0.5*(sign(dot(unph, n))+1)
perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
u_upwind = lambda q: Upwind('+')*q('+') + Upwind('-')*q('-')


def uadv_eq(w):
    return(-inner(perp(grad(inner(w, perp(unph)))), unph)*dx
           - inner(jump(inner(w, perp(unph)), n), perp_u_upwind(unph))*(dS)
           # - inner(inner(w, perp(unph))* n, unph) * ( ds_t + ds_b )
           - 0.5 * inner(unph, unph) * div(w) * dx
           # + 0.5 * inner(u_upwind(unph), u_upwind(unph)) * jump(w, n) * dS_h
           )


def u_eqn(w, gammar):
    return (inner(w, unp1 - un)*dx + dT * (
            uadv_eq(w)
            - c_p*div(w*thetanph)*Pinph*dx
            + c_p*jump(thetanph*w, n)*lambdarnp1('+')*dS_h
            + c_p*inner(thetanph*w, n)*lambdarnp1*(ds_t + ds_b)
            + c_p*jump(thetanph*w, n)*(0.5*(Pinph('+') + Pinph('-')))*dS_v
            # + c_p * inner(thetanph * w, n) * Pinph * (ds_v)
            + gammar('+')*jump(unph, n)*dS_h
            + gammar*inner(unph, n)*(ds_t + ds_b)
            + g * inner(w, zvec)*dx)
            )


dS = dS_h + dS_v


def rho_eqn(phi):
    return (phi*(rhonp1 - rhon)*dx
            + dT * (-inner(grad(phi), rhonph*unph)*dx
                    + (phi('+') - phi('-'))*(unn('+')*rhonph('+') - unn('-')*rhonph('-'))*dS
                    # + dot(phi*unph,n) *ds_v
                    )
            )


def theta_eqn(chi):
    return (chi*(thetanp1 - thetan)*dx
            + dT * (inner(chi*unph, grad(thetanph))*dx
                    + (chi('+') - chi('-'))* (unn('+')*thetanph('+') - unn('-')*thetanph('-'))*dS
                    - dot(chi('+')*thetanph('+')*unph('+'),  n('+'))*dS - inner(chi('-')*thetanph('-')*unph('-'), n('-'))*dS
                    # + dot(unph*chi,n)*thetanph * (ds_v + ds_t + ds_b)
                    # - inner(chi*thetanph * unph, n)* (ds_v +  ds_t + ds_b)
                    )
            )


# set up test functions and the nonlinear problem
w, phi, chi, gammar = TestFunctions(W)
# a = Constant(10000.0)
eqn = u_eqn(w, gammar) + theta_eqn(chi) + rho_eqn(phi) # + gamma * rho_eqn(div(w))
nprob = NonlinearVariationalProblem(eqn, Unp1)

# multigrid solver
sparameters_mg = {
        "snes_monitor": None,
        "mat_type": "aij",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "python",
        'mg_levels_pc_python_type': 'firedrake.ASMStarPC',
        "mg_levels_pc_star_sub_pc_type": "lu",
        'mg_levels_pc_star_dims': '0',
        'mg_levels_pc_star_sub_pc_factor_mat_solver_type' : 'mumps',
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    }

nsolver = NonlinearVariationalSolver(nprob, solver_parameters=sparameters_mg)

# start with these initial guesses
un.assign(u0)
rhon.assign(rho0)
thetan.assign(theta0)
lambdarn.assign(lambdar0)

print("rho max min", rhon.dat.data.max(),  rhon.dat.data.min())
print("theta max min", thetan.dat.data.max(), thetan.dat.data.min())
print("lambda max min", lambdarn.dat.data.max(), lambdarn.dat.data.min())

# initial guess for Unp1 is Un
Unp1.assign(Un)

name = "../../Results/compEuler/gravity-wave/flat/gravity_flat"
file_out = File(name+'.pvd')

rhon_pert = Function(Vp)
thetan_pert = Function(Vt)
rhon_pert.assign(rhon - rho_b)
thetan_pert.assign(thetan - theta_b)

file_out.write(un, rhon_pert, thetan_pert)

dt = 12.
dumpt = 120.
tdump = 0.
dT.assign(dt)
tmax = 6000.

print('tmax', tmax, 'dt', dt)
t = 0.


while t < tmax - 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()

    Un.assign(Unp1)

    rhon_pert.assign(rhon - rho_b)
    thetan_pert.assign(thetan - theta_b)

    print("rho max min pert", rhon_pert.dat.data.max(), rhon_pert.dat.data.min())
    print("theta max min pert", thetan_pert.dat.data.max(), thetan_pert.dat.data.min())

    if tdump > dumpt - dt*0.5:
        file_out.write(un, rhon_pert, thetan_pert)

        # file_gw.write(un, rhon, thetan, lambdarn)
        # file2.write(un_pert, rhon_pert, thetan_pert)
        tdump -= dumpt
