
import firedrake as fd
from firedrake.petsc import PETSc

from galewsky_utils import *
from lonlat_plot_utils import *

from firedrake import *
from TimeSteppingClass import *
import timeit

R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant


# function will carry solution in a rectangualr lonlat mesh
V_rect = get_V_ll_rect()

D_ll = Function(V_rect, name="D")
u_ll = Function(V_rect, name="u magnitude")
q_ll = Function(V_rect, name="relative vorticity") 
pot_q_ll = Function(V_rect, name="potential vorticity") 
courant_ll = Function(V_rect, name="courant number") 



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

mesh_sphere = IcosahedralSphereMesh(radius=R0,
                            degree=1,
                            refinement_level=5,
                            distribution_parameters = distribution_parameters)
x = SpatialCoordinate(mesh_sphere)
mesh_sphere.init_cell_orientations(x)

#help function space for plotting latlon
VS = FunctionSpace(mesh_sphere, "CG", degree=2)



u_expr = velocity_expression(*x)


# Initial depth
D_expr, D_pert = depth_expression(*x) #D_expr already inludes perturbation
bexpr = 0

file_pert = File("Results/Galewsky/D_pert.pvd")
D_pert = Function(VS, name = "D perturbation").interpolate(D_pert)
D_pert_ll = Function(V_rect, name = "D_pert")
sphere_to_latlongrect(D_pert, D_pert_ll)
file_pert.write(D_pert_ll)


# set times
T = 6*24*60*60
dt = 100.
dtc = Constant(dt)
t_inner = 0.
dt_inner = dt/10.
dt_inner_c = Constant(dt_inner)


#set up class, sets up all necessery solvers
SWE_stepper_1 = SWEWithProjection(mesh_sphere, dtc/2, u_expr, D_expr, H)
SWE_stepper_2 = SWEWithProjection(mesh_sphere, dtc, u_expr, D_expr, H, second_order=True)

SWE_stepper_2.compute_Courant()
SWE_stepper_2.compute_vorticity()
out_file = File("Results/Galewsky/Galewsky.pvd")
out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.un, SWE_stepper_2.D_pert, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)


out_file_ll = File("Results/Galewsky/Galewsky_lonlat.pvd")
sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
out_file_ll.write(D_ll, u_ll, q_ll, pot_q_ll, courant_ll)

t = 0.0
step = 0
dumpt = 36*dt
#dumpt = 4*3600 # every 4 hours
tdump = 0. 
output_freq = 20

#for step in ProgressBar(f'average forward').iter(range(ns)):
energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
with open("Results/Galewsky/energies.txt","w") as file_energies:
    file_energies.write(str(energy)+'\n')
with open("Results/Galewsky/enstrophies.txt","w") as file:
    file.write(str(enstrophy)+'\n')
with open("Results/Galewsky/divl2.txt","w") as file:
    file.write(str(div_l2)+'\n')

'''
with open("Results/Galewsky/energies_err.txt","w") as file_energies:
    file_energies.write(str(energy_err)+'\n')
with open("Results/Galewsky/enstrophies_err.txt","w") as file:
    file.write(str(enstrophy_err)+'\n')
with open("Results/Galewsky/divl2_err.txt","w") as file:
    file.write(str(divl2_err)+'\n')
'''  

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

    '''
    stop = timeit.default_timer()
    time = stop - start
    print('Time/step: ', time) 
    with open("runtimes.txt","a") as file_times:
        file_times.write(str(time)+'\n')
    '''

    step += 1
    t += dt
    tdump+=dt
    print("t = ", t)

    if tdump > dumpt - dt*0.5:

        print("t =", t)

        SWE_stepper_2.compute_Courant()
        qn = SWE_stepper_2.compute_vorticity()
        out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.un, SWE_stepper_2.D_pert, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

        sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
        sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
        sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
        sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
        sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
        out_file_ll.write(D_ll, u_ll, q_ll, pot_q_ll, courant_ll)

        energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
        with open("Results/Galewsky/energies.txt","a") as file_energies:
            file_energies.write(str(energy)+'\n')
        with open("Results/Galewsky/enstrophies.txt","a") as file_enstrophies:
            file_enstrophies.write(str(enstrophy)+'\n')
        with open("Results/Galewsky/divl2.txt","a") as file_div:
            file_div.write(str(div_l2)+'\n')
        
        '''
        with open("Results/Galewsky/energies_err.txt","a") as file_energies:
            file_energies.write(str(energy_err)+'\n')
        with open("Results/Galewsky/enstrophies_err.txt","a") as file:
            file.write(str(enstrophy_err)+'\n')
        with open("Results/Galewsky/divl2_err.txt","a") as file:
            file.write(str(divl2_err)+'\n')
        '''

        tdump -= dumpt
   