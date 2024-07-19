from firedrake import *
from TimeSteppingClass import *
from lonlat_plot_utils import *

R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant

V_rect = get_V_ll_rect()

D_ll = Function(V_rect, name="D")
Dpb_ll= Function(V_rect, name="D+b")
u_ll = Function(V_rect, name="u magnitude")
q_ll = Function(V_rect, name="relative vorticity") 
pot_q_ll = Function(V_rect, name="potential vorticity") 
courant_ll = Function(V_rect, name="courant number") 

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

mesh_sphere = IcosahedralSphereMesh(radius=R0,
                            degree=1,
                            refinement_level=5,
                            distribution_parameters = distribution_parameters)
x = SpatialCoordinate(mesh_sphere)
mesh_sphere.init_cell_orientations(x)

#help function space for plotting latlon
VS = FunctionSpace(mesh_sphere, "CG", degree=2)


# Initial velocity (solid body rotation)
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])


# Initial depth
D_expr = H - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g


#ALTERNATIVE
#x_c = as_vector([1., 0., 0.])
#F_0 = Constant(3.)
#l_0 = Constant(0.25)
#def dist_sphere(x, x_c):
#    return acos(dot(x/R0,x_c))

#F_theta = F_0*exp(-dist_sphere(x,x_c)**2/l_0**2)
#D_expr = conditional(dist_sphere(x,x_c) > 0.5, 0., F_theta)

#Topography
rl = pi/9.0
lambda_x = atan2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = min_value(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)


# set times
T = 15*86400.
dt = 3600. # one hour
dtc = Constant(dt)



#set up class, sets up all necessery solvers
SWE_stepper_1 = SWEWithProjection(mesh_sphere, dtc/2, u_expr, D_expr, H, 
                                  bexpr=bexpr, n_adv_cycles = 3, slate_inv=True)
SWE_stepper_2 = SWEWithProjection(mesh_sphere, dtc, u_expr, D_expr, H, 
                                  bexpr=bexpr, second_order=True, n_adv_cycles = 3,
                                  slate_inv=True)

SWE_stepper_2.compute_Courant()
SWE_stepper_2.compute_vorticity()
out_file = File("Results/Williamson5/Williamson5.pvd")
out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

out_file_ll = File("Results/Williamson5/Williamson5_ll.pvd")
sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
sphere_to_latlongrect(SWE_stepper_2.Dplusb, Dpb_ll)
sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
out_file_ll.write(D_ll, Dpb_ll, u_ll, q_ll, pot_q_ll, courant_ll)

t = 0.0
step = 0
dumpt = 86400./4 #=21600
tdump = 0.

#for step in ProgressBar(f'average forward').iter(range(ns)):
energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
with open("Results/Williamson5/energies.txt","w") as file_energies:
    file_energies.write(str(energy)+'\n')
with open("Results/Williamson5/enstrophies.txt","w") as file:
    file.write(str(enstrophy)+'\n')
with open("Results/Williamson5/divl2.txt","w") as file:
    file.write(str(div_l2)+'\n')
   


while t < T - 0.5*dt:

    print("t = ", t)
 
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

    step += 1
    t += dt
    tdump += dt

    if tdump > dumpt - dt*0.5:

        print("t =", t)

        SWE_stepper_2.compute_Courant()
        qn = SWE_stepper_2.compute_vorticity()
        out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.Dplusb,SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

        sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
        sphere_to_latlongrect(SWE_stepper_2.Dplusb, Dpb_ll)
        sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
        sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
        sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
        sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
        out_file_ll.write(D_ll, Dpb_ll, u_ll, q_ll, pot_q_ll, courant_ll)

        energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
        with open("Results/Williamson5/energies.txt","a") as file_energies:
            file_energies.write(str(energy)+'\n')
        with open("Results/Williamson5/enstrophies.txt","a") as file_enstrophies:
            file_enstrophies.write(str(enstrophy)+'\n')
        with open("Results/Williamson5/divl2.txt","a") as file_div:
            file_div.write(str(div_l2)+'\n')
        
        tdump -= dumpt
   