from firedrake import *

with CheckpointFile("checkpointNHMW0.h5", 'r') as file0:
    with CheckpointFile("checkpointNHMW1.h5", 'r') as file1:
        with CheckpointFile("checkpointNHMW2.h5", 'r') as file2:
            with CheckpointFile("checkpointNHMW3.h5", 'r') as file3:
                mesh0 = file0.load_mesh("mesh")
                mesh1 = file1.load_mesh("mesh")
                mesh2 = file2.load_mesh("mesh")
                mesh3 = file3.load_mesh("mesh")
                print("meshes loaded")

                for i in range(15):
                    U0 = file0.load_function(mesh0, "Un", idx=i)
                    u0, _,theta0, _ = U0.subfunctions
                    print("k=0, norm for function ", i, "is ", assemble(inner(u0,u0)*dx))

                    U1 = file1.load_function(mesh1, "Un", idx=i)
                    u1, _,theta1, _ = split(U1)
                    print("k=1, norm for function ", i, "is ", assemble(inner(u1,u1)*dx))

                    U2 = file2.load_function(mesh2, "Un", idx=i)
                    u2, _,theta2, _ = split(U2)
                    print("k=2, norm for function ", i, "is ", assemble(inner(u2,u2)*dx))

                    U3 = file3.load_function(mesh3, "Un", idx=i)
                    u3, _,theta3, _ = split(U3)
                    print("k=3, norm for function ", i, "is ", assemble(inner(u3,u3)*dx))

                    u3, _,theta3, _ = U3.subfunctions

                    # First, grab the mesh.

                    V = u3.function_space()


                    # Now make the VectorFunctionSpace corresponding to V.
                    #W = VectorFunctionSpace(mesh3, V.ufl_element())

                    f = Function(V).interpolate(u0)

# Next, interpolate the coordinates onto the nodes of W.
#X = interpolate(m.coordinates, W)

# Make an output function.
#f = Function(V)

# Use the external data function to interpolate the values of f.
#f.dat.data[:] = mydata(X.dat.data_ro)


                    V = u3.function_space()
                    g = Function(functionspaceimpl.WithGeometry.create(u0.function_space(), mesh0), val=u0.topological)
                    #u0 = Function(V).interpolate(u0)
                    #print(assemble(inner(u0-u3,u0-u3)*dx))

                    #print("error for time step ", i*50, "is ", assemble(inner(u_new - u_old,u_new-u_old)*dx))
                    #print(assemble(inner(u_old,u_old)*dx))
                    #print(norm(u_new - u_old))
