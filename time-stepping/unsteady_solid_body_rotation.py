from firedrake import *
from TimeSteppingClass import *
import timeit

'''
CONVERGENCE TEST
Lauter test, Example 3 in https://doi.org/10.1016/j.jcp.2005.04.022
'''

R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
Omega_vec = Omega*as_vector([0.,0.,1.])
g = Constant(9.8)  # Gravitational constant

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
refinement_level_list= [6] #[3,4,5,6]
n_ref = len(refinement_level_list)

for i in range(n_ref):
    refinement_level=refinement_level_list[i]
    mesh = IcosahedralSphereMesh(radius=R0,
                                degree=3, #TODO increase
                                refinement_level=refinement_level,
                                distribution_parameters = distribution_parameters)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    n = CellNormal(mesh)

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


    #velocity space
    element = FiniteElement("BDM", triangle, degree=2)
    V1 = FunctionSpace(mesh, element)
    #space for height:
    V2 = FunctionSpace(mesh, "DG", 1)
    file_exact = File("Results/LauterTest/exact_solution"+str(refinement_level)+ ".pvd")
    u_ex_0, D_ex_0 = exact_sol(t)
    u_ex = Function(V1, name = "u").interpolate(u_ex_0)
    D_ex = Function(V2, name = "D").interpolate(D_ex_0)
    file_exact.write(u_ex, D_ex)


    # set times
    #T=1.
    T = 2*86400.
    dt = 50.
    dtc = Constant(dt)

    #set up class, sets up all necessery solvers
    SWE_stepper_1 = SWEWithProjection(mesh, dtc/2, u_expr, D_expr, H, 
                                      bexpr=bexpr, n_adv_cycles=1, slate_inv = True)
    SWE_stepper_2 = SWEWithProjection(mesh, dtc, u_expr, D_expr, H, 
                                      bexpr=bexpr, second_order=True, n_adv_cycles=1, slate_inv = True)

    SWE_stepper_2.compute_Courant()
    SWE_stepper_2.compute_vorticity()
    out_file = File("Results/LauterTest/unsteady_body_rot"+str(refinement_level)+ ".pvd")
    out_file.write(SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

    t = 0.0
    step = 0
    output_freq = 100.

    #for step in ProgressBar(f'average forward').iter(range(ns)):
    u_error = norm(SWE_stepper_2.un - u_ex_0)/uref
    D_error = norm(SWE_stepper_2.Dplusb - D_ex_0)/dref

    with open("Results/LauterTest/uerrors"+str(refinement_level)+".txt", 'w') as ufile:
        ufile.write(str(u_error)+'\n')
    with open("Results/LauterTest/Derrors"+str(refinement_level)+".txt", 'w') as Dfile:
        Dfile.write(str(D_error)+'\n')

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

        if step % output_freq == 0:

            print("t =", t)

            SWE_stepper_2.compute_Courant()
            qn = SWE_stepper_2.compute_vorticity()
            out_file.write(SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

            '''
            energy, enstrophy, div_l2 = SWE_stepper_2.compute_phys_quant()
            with open("Results/LauterTest/"+str(refinement_level)+"/energies.txt","a") as file_energies:
                file_energies.write(str(energy)+'\n')
            with open("Results/LauterTest/"+str(refinement_level)+"/enstrophies.txt","a") as file_enstrophies:
                file_enstrophies.write(str(enstrophy)+'\n')
            with open("Results/LauterTest/"+str(refinement_level)+"/divl2.txt","a") as file_div:
                file_div.write(str(div_l2)+'\n')
            '''
            u_exact, D_exact = exact_sol(t)
            u_ex.project(u_exact)
            D_ex.project(D_exact)
            file_exact.write(u_ex, D_ex)
            u_error = norm(SWE_stepper_2.un - u_exact)
            D_error = norm(SWE_stepper_2.Dplusb - D_exact)
            #u_error = sqrt(assemble(inner(SWE_stepper_2.un - u_exact, SWE_stepper_2.un - u_exact)*dx))/sqrt(assemble(inner(u_exact, u_exact)*dx))
            #D_error = sqrt(assemble(inner(SWE_stepper_2.Dplusb - D_exact, SWE_stepper_2.Dplusb - D_exact)*dx))/sqrt(assemble(inner(D_exact, D_exact)*dx))

            #u_ex.interpolate(u_exact)
            #D_ex.interpolate(D_exact)
            #file_exact.write(u_ex, D_ex)
            
            #u_error, D_error = SWE_stepper_2.compute_error_sbr()
            with open("Results/LauterTest/uerrors"+str(refinement_level)+".txt", 'a') as ufile:
                ufile.write(str(u_error)+'\n')
            with open("Results/LauterTest/Derrors"+str(refinement_level)+".txt", 'a') as Dfile:
                Dfile.write(str(D_error)+'\n')

        
