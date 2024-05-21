from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

mesh_sizes = np.array([1/10, 1/20, 1/40, 1/80, 1/125, 1/160]) 
uerrors = np.zeros(len(mesh_sizes))
thetaerrors = np.zeros(len(mesh_sizes))

uc = 100.
vc = 100.
L = 10000.
H = 10000.

'''
ORDER OF SPACES DIFFERENT ON MY MACHINE AND CONNORS MACHINE
condier when analysing convergence
currently run on Connors machine is nx=160
'''

index =0

with CheckpointFile("checkpointVortex10.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =16)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index+=1

with CheckpointFile("checkpointVortex20.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =16)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index +=1

with CheckpointFile("checkpointVortex40.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =16)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))
    
index +=1


with CheckpointFile("checkpointVortex80.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =16)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))
    
index +=1


with CheckpointFile("checkpointVortex125.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =128)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index+=1

with CheckpointFile("checkpointVortex160.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =128)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

'''
#uerrors = np.array([0.02345874, 0.00715841, 0.00145645, 0.00024]) 
#thetaerrors = np.array([1.58243427e-02, 3.73220993e-03, 6.18298058e-04, 1.23099994e-04])
uerrors = [0.02345874, 0.00715841, 0.00145645, 0.00024, 0.07708535] 
thetaerrors = [1.58243427e-02, 3.73220993e-03, 6.18298058e-04, 1.23099994e-04, 7.81958405e-05]
'''

print(uerrors, thetaerrors)   

hpower = mesh_sizes**2

fig, axes = plt.subplots()
axes.set_title("Loglog plot of velocity errors")
plt.loglog(mesh_sizes, uerrors, color = "blue", label = "u-error", marker = ".")
plt.loglog(mesh_sizes, 1e-1*hpower, color = "orange", label = "h^2", marker = ".")
axes.legend()
fig.savefig("uerrors.png")
plt.show()

fig, axes = plt.subplots()
axes.set_title("Loglog plot of theta errors")
plt.loglog(mesh_sizes, thetaerrors, color = "blue", label = "theta-error", marker = ".")
plt.loglog(mesh_sizes, hpower, color = "orange", label = "h^2", marker = ".")
axes.legend()
fig.savefig("thetaerrors.png")
plt.show()


'''
nx = 128, dt=0.1
nx = 64, dt = 0.25
nx = 32, dt=0.5
'''