
import firedrake as fd
from firedrake.petsc import PETSc

from galewsky_utils import *

from firedrake import *
from TimeSteppingClass import *
import timeit


R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant
umax = 80.
Umax = fd.Constant(umax)

h0 = 10e3
H0 = fd.Constant(h0)

theta0 = pi/7.
theta1 = pi/2. - theta0
en = np.exp(-4./((theta1-theta0)**2))

alpha = fd.Constant(1/3.)
beta = fd.Constant(1/15.)
hhat = fd.Constant(120)
theta2 = fd.Constant(pi/4.)


distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

mesh = IcosahedralSphereMesh(radius=R0,
                            degree=1,
                            refinement_level=5,
                            distribution_parameters = distribution_parameters)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)


u_expr = velocity_expression(*x)


# Initial depth
D_expr = depth_expression(*x)
bexpr = 0


# set times
T = 6*24*60*60
dt = 100.
dtc = Constant(dt)
t_inner = 0.
dt_inner = dt/10.
dt_inner_c = Constant(dt_inner)


#set up class, sets up all necessery solvers
SWE_stepper_1 = SWEWithProjection(mesh, dtc/2, u_expr, D_expr, H)
SWE_stepper_2 = SWEWithProjection(mesh, dtc, u_expr, D_expr, H, second_order=True)

SWE_stepper_2.compute_Courant()
SWE_stepper_2.compute_vorticity()
out_file = File("Results/Galewsky.pvd")
out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

t = 0.0
step = 0
output_freq = 20

#for step in ProgressBar(f'average forward').iter(range(ns)):
energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
with open("energies.txt","a") as file_energies:
    file_energies.write(str(energy)+'\n')
with open("enstrophies.txt","a") as file:
    file.write(str(enstrophy)+'\n')
with open("divl2.txt","a") as file:
    file.write(str(div_l2)+'\n')
   

while t < T - 0.5*dt:

    start = timeit.default_timer()
 
    SWE_stepper_1.un.assign(SWE_stepper_2.un)
    SWE_stepper_1.Dn.assign(SWE_stepper_2.Dn)

    #first order, compute unph with delt/2, computes unph saved in SWE_stepper_1.un
    SWE_stepper_1.advection_SSPRK3()
    SWE_stepper_1.projection_step()

    SWE_stepper_2.ubar.assign(SWE_stepper_1.un) # solution from first order scheme
    #solve 2 equations for second step 
    SWE_stepper_2.second_order_1st_step()
    SWE_stepper_2.advection_SSPRK3()
    SWE_stepper_2.projection_step()

    stop = timeit.default_timer()
    time = stop - start
    print('Time/step: ', time) 
    with open("runtimes.txt","a") as file_times:
        file_times.write(str(time)+'\n')
   

    step += 1
    t += dt

    if step % output_freq == 0:

        print("t =", t)

        SWE_stepper_2.compute_Courant()
        qn = SWE_stepper_2.compute_vorticity()
        out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

        energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
        with open("energies.txt","a") as file_energies:
            file_energies.write(str(energy)+'\n')
        with open("enstrophies.txt","a") as file_enstrophies:
            file_enstrophies.write(str(enstrophy)+'\n')
        with open("divl2.txt","a") as file_div:
            file_div.write(str(div_l2)+'\n')
   