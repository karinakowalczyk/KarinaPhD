from firedrake import *


with CheckpointFile("mountain_nh.h5", 'r') as fileold:
    with CheckpointFile("checkpointNHMW.h5", 'r') as filenew:
        with CheckpointFile("background_fcts.h5", 'r') as fileb:
            with CheckpointFile("backgroundfcts_new.h5", 'r') as filebnew:

                mesh_old = fileold.load_mesh("mesh")
                mesh_new = filenew.load_mesh("mesh")
                print("meshes loaded")

                thetab = fileb.load_function(mesh_old, "thatab")
                rhob = fileb.load_function(mesh_old, "rohb")

                thetab_new = filebnew.load_function(mesh_old, "thetab")
                rhob_new = filebnew.load_function(mesh_old, "rhob")

                print("error thata background is ", sqrt(assemble(inner(thetab_new - thetab, thetab_new-thetab)*dx)))
                print("error rho background is ", sqrt(assemble(inner(rhob_new - rhob, rhob_new-rhob)*dx)))
                    
                for i in range(37):
                    U_old = fileold.load_function(mesh_old, "Un", idx=i)
                    u_old, _,theta_old = split(U_old)

                    deltheta_old = theta_old - thetab
                    #print("norm for function ", i, "is ", assemble(inner(u_old,u_old)*dx))

                    U_new = filenew.load_function(mesh_old, "Un", idx=i)
                    u_new, _, theta_new, _ = split(U_new)
                    deltheta_new = theta_new - thetab_new
                    print("error for time step ", i*50*5, "is ", sqrt(assemble(inner(deltheta_new - deltheta_old,deltheta_new-deltheta_old)*dx)))
                    #print(assemble(inner(u_old,u_old)*dx))
                    #print(norm(u_new - u_old))

    
        
'''
# set mesh parameters
L = 144000.
H = 35000.  # Height position of the model top
delx = 400*2
delz = 250*2
nlayers = H/delz  # horizontal layers
columns = L/delx  # number of columns

# build mesh
m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)


#build spaces
horizontal_degree = 1
vertical_degree = 1

# horizontal base spaces
cell = mesh._base_mesh.ufl_cell().cellname()
S1 = FiniteElement("CG", cell, horizontal_degree + 1)  # EDIT: family replaced by CG (was called with RT before)
S2 = FiniteElement("DG", cell, horizontal_degree, variant="equispaced")

# vertical base spaces
T0 = FiniteElement("CG", interval, vertical_degree + 1, variant="equispaced")
T1 = FiniteElement("DG", interval, vertical_degree, variant="equispaced")

# trace base space
Tlinear = FiniteElement("CG", interval, 1)

# build spaces V2, V3, Vt
V2h_elt = HDiv(TensorProductElement(S1, T1))
V2t_elt = TensorProductElement(S2, T0)
V3_elt = TensorProductElement(S2, T1)
V2v_elt = HDiv(V2t_elt)

V2v_elt_Broken = BrokenElement(HDiv(V2t_elt))
V2_elt = EnrichedElement(V2h_elt, V2v_elt_Broken)
VT_elt = TensorProductElement(S2, Tlinear)

remapped = WithMapping(V2_elt, "identity")

V0 = FunctionSpace(mesh, remapped, name="new_velocity")
V1 = FunctionSpace(mesh, V3_elt, name="DG")  # pressure space
V2 = FunctionSpace(mesh, V2t_elt, name="Temp")
u0 = as_vector([10., 0.])

u = Function(V0).project(u0)
rho = Function(V1).interpolate(Constant(1.))

with CheckpointFile("exampleNEW.h5", 'w') as afile:
    afile.save_function(u)
    afile.save_function(rho)
'''