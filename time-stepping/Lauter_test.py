from firedrake import *
from TimeSteppingClass import *
import timeit
from lonlat_plot_utils import *

'''
Lauter test, Example 3 in https://doi.org/10.1016/j.jcp.2005.04.022
'''

R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
Omega_vec = Omega*as_vector([0.,0.,1.])
g = Constant(9.8)  # Gravitational constant

V_rect = get_V_ll_rect()

D_ll = Function(V_rect, name="D")
Dpb_ll= Function(V_rect, name="D+b")
u_ll = Function(V_rect, name="u magnitude")
q_ll = Function(V_rect, name="relative vorticity") 
pot_q_ll = Function(V_rect, name="potential vorticity") 
courant_ll = Function(V_rect, name="courant number") 

#l2 norm of exact solution as reference, computed once on a mesh with ref_level=7
uref = 712008703.4132433 
dref = 298004433314.0549


t =0.

e1 = as_vector([1.,0.,0.])
e2 = as_vector([0.,1.,0.])
e3 = as_vector([0.,0.,1.])
b1 = cos(Omega*t)*e1 + sin(Omega*t)*e2
b2 = -sin(Omega*t)*e1 + cos(Omega*t)*e2
b3 = e3

alpha = pi/4
c_vec = -sin(alpha)*e1 + cos(alpha)*e3

u0 = (2*pi*R0/12)/86400.
#u0 = 20.
H= 133681./g

def compute_ref():
    refinement_level=7
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    mesh = IcosahedralSphereMesh(radius=R0,
                                degree=3, #TODO increase
                                refinement_level=refinement_level,
                                distribution_parameters = distribution_parameters)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    phi_t = dot(c_vec, b1)*e1 +dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3 

    # needed because mesh isn't actually on the sphere, so n isn't exact 
    r = (x[0]**2+x[1]**2+x[2]**2)**0.5
    k = as_vector(x)/r
    z = x[2]/r*R0

    phi_t_dot_n = dot(phi_t, k)
    #= (1/R0)*(-sin(alpha)*x[0]+cos(alpha)*x[2]) 
    u_expr = u0 * cross(phi_t, k)
    D_expr = H - (1/(2*g))*((u0*phi_t_dot_n+ z*Omega)**2 ) + (1/(2*g))*(z*Omega)**2
    uref = norm(u_expr)
    Dref = norm(D_expr)
    return uref, Dref

if uref == 0:
    print("Computing reference functions")
    uref, dref = compute_ref()
   
print("uref, dref =", uref, dref)

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}


mesh_sphere = IcosahedralSphereMesh(radius=R0,
                            degree=1, #TODO increase
                            refinement_level=5,
                            distribution_parameters = distribution_parameters)
x = SpatialCoordinate(mesh_sphere)
mesh_sphere.init_cell_orientations(x)
n = CellNormal(mesh_sphere)

VS = FunctionSpace(mesh_sphere, "CG", degree=2)

t =0.

#e1 = as_vector([1.,0.,0.])
#e2 = as_vector([0.,1.,0.])
#e3 = as_vector([0.,0.,1.])
b1 = cos(Omega*t)*e1 + sin(Omega*t)*e2
b2 = -sin(Omega*t)*e1 + cos(Omega*t)*e2
b3 = e3


alpha = pi/4
c_vec = -sin(alpha)*e1 + cos(alpha)*e3

#phi = dot(c_vec, b1)*e1 +dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3 
#phi_dot_n = dot(phi, n)

u0 = (2*pi*R0/12)/86400.
#u0 = 20.
H= 133681./g
phi_t = dot(c_vec, b1)*e1 +dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3 

# needed because mesh isn't actually on the sphere, so n isn't exact 
r = (x[0]**2+x[1]**2+x[2]**2)**0.5
k = as_vector(x)/r
z = x[2]/r*R0

phi_t_dot_n = dot(phi_t, k)
#= (1/R0)*(-sin(alpha)*x[0]+cos(alpha)*x[2]) 
u_expr = u0 * cross(phi_t, k)
D_expr = H - (1/(2*g))*((u0*phi_t_dot_n+ z*Omega)**2 ) + (1/(2*g))*(z*Omega)**2

#u_expr = as_vector([-u0*x[1]/R0, u0*x[0]/R0, 0.0])
# Initial depth
#D_expr = H - ((R0 * Omega * u0 + u0*u0/2.0)*(x[2]*x[2]/(R0*R0)))/g
k2 = 0.
bexpr = (1/(2*g))*(z*Omega)**2 + k2
#bexpr =0.
#u_expr = (u0/R0)*as_vector([-cos(alpha)*x[1], cos(alpha)*x[0]+ sin(alpha)*x[2], -sin(alpha)*x[1]])

def exact_sol(t):
    b1 = cos(Omega*t)*e1 + sin(Omega*t)*e2
    b2 = -sin(Omega*t)*e1 + cos(Omega*t)*e2
    b3 = e3
    c_vec = -sin(alpha)*e1 + cos(alpha)*e3
    phi_t = dot(c_vec, b1)*e1 +dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3 
    phi_t_dot_n = dot(phi_t, k)
    u_exact = u0 * cross(phi_t, k)
    D_exact = H - (1/(2*g))*(u0*phi_t_dot_n+ z*Omega)**2 + (1/(2*g))*(z*Omega)**2
    return u_exact, D_exact


# set times
#T=1.
T = 12*86400.
dt = 400.
dtc = Constant(dt)

#set up class, sets up all necessery solvers
SWE_stepper_1 = SWEWithProjection(mesh_sphere, dtc/2, u_expr, D_expr, H, 
                                    bexpr=bexpr, n_adv_cycles=3, slate_inv = True)
SWE_stepper_2 = SWEWithProjection(mesh_sphere, dtc, u_expr, D_expr, H, 
                                    bexpr=bexpr, second_order=True, n_adv_cycles=3, slate_inv = True)

SWE_stepper_2.compute_Courant()
SWE_stepper_2.compute_vorticity()
out_file = File("Results/LauterTest/Lauter.pvd")
out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

out_file_ll = File("Results/LauterTest/Lauter_ll.pvd")
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

energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
with open("Results/LauterTest/energies.txt","w") as file_energies:
    file_energies.write(str(energy)+'\n')
with open("Results/LauterTest/enstrophies.txt","w") as file_enstrophies:
    file_enstrophies.write(str(enstrophy)+'\n')
with open("Results/LauterTest/divl2.txt","w") as file_div:
    file_div.write(str(div_l2)+'\n')
        



while t < T - 0.5*dt:
    
    print("time ", t)

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
        out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

        sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
        sphere_to_latlongrect(SWE_stepper_2.Dplusb, Dpb_ll)
        sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
        sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
        sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
        sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
        out_file_ll.write(D_ll, Dpb_ll, u_ll, q_ll, pot_q_ll, courant_ll)
        
        energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
        with open("Results/LauterTest/energies.txt","a") as file_energies:
            file_energies.write(str(energy)+'\n')
        with open("Results/LauterTest/enstrophies.txt","a") as file_enstrophies:
            file_enstrophies.write(str(enstrophy)+'\n')
        with open("Results/LauterTest/divl2.txt","a") as file_div:
            file_div.write(str(div_l2)+'\n')
        
        tdump -= dumpt
