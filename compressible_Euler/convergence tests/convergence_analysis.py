from firedrake import *


with CheckpointFile("checkpointVortex32.h5", 'r') as file:
            mesh = file.load_mesh("mesh")
            print("meshes loaded")
            U0 = file.load_function(mesh, "Un", idx =0)
            u0, rho0, theta0, _ = U0.subfunctions
            U100 = file.load_function(mesh, "Un", idx =200)
            u100, rho100, theta100, _ = U100.subfunctions
            print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
            print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
            print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))
            

'''
nx = 128, dt=0.1
nx = 64, dt = 0.25
nx = 32, dt=0.5
'''