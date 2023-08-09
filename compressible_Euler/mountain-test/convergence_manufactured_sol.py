import firedrake as fd
import sympy as sp

x, z = sp.symbols('x z')

Ax = 1.5
Az = 0.6
Bx = 1.
Bz = 0.9

sin = sp.sin
cos = sp.cos
pi = sp.pi
exp = sp.exp

ux = 0.1*exp(0.1*(cos(Ax*pi*x)+sin(Az*pi*z)))
uz = 0.3*exp(0.1*(sin(Bx*pi*x)+cos(Bz*pi*z)))

ugradu_x = ux*ux.diff(x) + uz*ux.diff(z)
ugradu_z = ux*uz.diff(x) + uz*uz.diff(z)

rho = 0.2*exp(0.1*(sin(pi*x)+cos(1.5*pi*z)))
theta = 0.2*exp(0.1*(cos(pi*x)+sin(2*pi*z)))

R_d = 287.  # Gas constant for dry air (J/kg/K)
kappa = 2.0/7.0  # R_d/c_p
p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)

Pi = (R_d*rho*theta/p_0)**(kappa/(1-kappa))
theta_Pi_diffx = theta*Pi.diff(x)
theta_Pi_diffz = theta*Pi.diff(z)
rhs_u_x = ugradu_x + theta_Pi_diffx
rhs_u_z = ugradu_z + theta_Pi_diffz

rhs_rho = 1+(rho*ux).diff(x) + (rho*uz).diff(z)

rhs_theta = 5 + ux*theta.diff(x) + uz*theta.diff(z)

sympy_input = rhs_rho


mesh = fd.UnitSquareMesh(10,10)
cos = fd.cos
sin = fd.sin
pi = fd.pi
exp = fd.exp
x, z = fd.SpatialCoordinate(mesh)
rhs_rho = eval(str(rhs_rho))
rhs_theta = eval(str(rhs_theta))
rhs_u_x = eval(str(rhs_u_x))
rhs_u_z = eval(str(rhs_u_z))
rhs_u = fd.as_vector([rhs_u_x, rhs_u_z])
print(fd.assemble(rhs_u_z*fd.dx))