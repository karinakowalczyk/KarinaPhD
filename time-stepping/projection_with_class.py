from firedrake import *
from TimeSteppingClass import *

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


T = 15*86400.
dt = 180.
dtc = Constant(dt)
t_inner = 0.
dt_inner = dt/10.
dt_inner_c = Constant(dt_inner)

SWE_stepper = TimeStepping(dtc)
t = 0.0
step = 0

output_freq = 20
out_file = File("Results/proj_solution.pvd")
out_file.write(SWE_stepper.Dn, SWE_stepper.un)


#for step in ProgressBar(f'average forward').iter(range(ns)):
while t < T - 0.5*dt:
    
    SWE_stepper.advection_SSPRK3()

    SWE_stepper.projection_step()

    step += 1
    t += dt

    if step % output_freq == 0:
        out_file.write(SWE_stepper.Dn, SWE_stepper.un)
        print("t =", t)
'''
        print(assemble((0.5*inner(un,un)*Dn + 0.5* g * (Dn+b)**2)*dx))

        assemble(Courant_num_form, tensor=Courant_num)
        courant_frac = Function(DG0).interpolate(Courant_num/Courant_denom)
        Courant.assign(courant_frac)

        qsolver.solve()
'''