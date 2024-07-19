from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from tools import build_spaces

mesh_sizes = np.array([1/10, 1/20, 1/40, 1/80, 1/125]) 
nx_list = [20, 40, 80, 160]
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


'''
DEFINE PERIODIC EXACT SOL
'''
uc = 100.
vc = 100.
L = 10000.
H = 10000.
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

nx =80
delx = 125.
m = PeriodicIntervalMesh(nx, L, distribution_parameters =
                            distribution_parameters)
mesh = ExtrudedMesh(m, nx, layer_height=delx, periodic=True, name='mesh')
x, y = SpatialCoordinate(mesh)

Vc = mesh.coordinates.function_space()
print(Vc)

#Vc = VectorFunctionSpace(mesh, "DG", 2)
xc = L/2
y_m = 300.
sigma_m = 500.
#y_s = y_m*exp(-((x - xc)/sigma_m)**2)
y_s = y_m*cos(2*pi*(x-xc)/L)
y_expr = y+y_s
new_coords = Function(Vc).interpolate(as_vector([x,y_expr]))
#mesh.coordinates.assign(new_coords)
x, y = SpatialCoordinate(mesh)
def exact_solution (x, y, t):

    xc = L/2 + t*uc
    yc = L/2 + t*vc

    #if xc >+ L:
    #    xc -=L
    #if yc>=L:
    #    yc -=L

    R = 0.4*L #radius of vortex
    distx = conditional(abs(x-xc)>L/2, L-abs(x-xc), abs(x-xc))
    disty = conditional(abs(y-yc)>L/2, L-abs(y-yc), abs(y-yc))
    diffx = x-xc
    diffy = y-yc
    r = sqrt(distx*distx + disty*disty)/R
    #r = conditional(abs(x-xc) <= L/2, sqrt(diffx*diffx + diffy*diffy)/R, sqrt((L-abs(x-xc))**2 + (L-abs(y-yc))**2)/R)
    print(type(r))
    #phi = atan2(x-xc, y - yc)

    rhoc = 1.
    rhob = conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6)

    uth = (1024 * (1.0 - r)**6 * (r)**6)
    #phi = atan2(x-xc, y-yc)
    signx = conditional(x>xc,-1, +1 )
    signy = conditional(y>yc,-1, +1 )
    diffx = conditional(abs(x-xc) <= L/2, x - xc, signx*(L-abs(x-xc)))
    diffy = conditional(abs(y-yc) <= L/2, y - yc, signy*(L-abs(y-yc)))
    #diffx = conditional(abs(x-xc) <= L/2, x - xc, L-abs(x-xc))
    #diffy = conditional(abs(y-yc) <= L/2, y - yc, L-abs(y-yc))
    ux = conditional(r>=1, uc, uc - uth * diffy/(r*R))
    
    uy = conditional(r>=1, vc, vc + uth * diffx/(r*R))
    u0 = as_vector([ux,uy])
    return u0, rhob

cell = mesh._base_mesh.ufl_cell().cellname()
CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = HDiv(TensorProductElement(CG_1, DG_0))
P0P1 = HDiv(TensorProductElement(DG_0, CG_1))
elt = P1P0 + P0P1
V, _, V1, _, _ = build_spaces(mesh, 1, 1)
u_out = Function(V)
rho_out = Function(V1)
'''
file_exact = File("../Results/convergence/vortex_exact.pvd")
ts = [0, 25., 50., 100.]
for t in ts:
    print(t)
    u, rho = exact_solution(x, y, t)
    
    u_out.project(u)
    rho_out.interpolate(rho)

    file_exact.write(u_out, rho_out)
'''

'''
index =0

file_test = File("test_checkpointed.pvd")

with CheckpointFile("checkpointVortex10.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    x,y = SpatialCoordinate(mesh)
    print("meshes loaded")
    #U0 = file.load_function(mesh, "Un", idx =0)
    #u0, rho0, theta0, _ = U0.subfunctions
    #U100 = file.load_function(mesh, "Un", idx = 4)
    #u100, rho100, theta100, _ = U100.subfunctions
    u0 = exact_solution(x, y, 25.)
    U100 = file.load_function(mesh, "Un", idx = 1)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    print(uerror)
    #thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    #thetaerrors[index] = thetaerror
    #print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    #print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    #print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index+=1

with CheckpointFile("checkpointVortex20.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    x,y = SpatialCoordinate(mesh)
    print("meshes loaded")
    #U0 = file.load_function(mesh, "Un", idx =0)
    #u0, rho0, theta0, _ = U0.subfunctions
    #U100 = file.load_function(mesh, "Un", idx = 4)
    #u100, rho100, theta100, _ = U100.subfunctions
    u0 = exact_solution(x, y, 25.)
    U100 = file.load_function(mesh, "Un", idx = 1)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    print(uerror)
    #thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    #thetaerrors[index] = thetaerror
    #print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    #print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    #print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index +=1

with CheckpointFile("checkpointVortex40.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    x,y = SpatialCoordinate(mesh)
    print("meshes loaded")
    #U0 = file.load_function(mesh, "Un", idx =0)
    #u0, rho0, theta0, _ = U0.subfunctions
    #U100 = file.load_function(mesh, "Un", idx = 4)
    #u100, rho100, theta100, _ = U100.subfunctions
    u0 = exact_solution(x, y, 25.)
    U100 = file.load_function(mesh, "Un", idx = 1)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    print(uerror)
    #thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    #thetaerrors[index] = thetaerror
    #print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    #print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    #print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

    
index +=1


with CheckpointFile("checkpointVortex80.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    x,y = SpatialCoordinate(mesh)
    print("meshes loaded")
    #U0 = file.load_function(mesh, "Un", idx =0)
    #u0, rho0, theta0, _ = U0.subfunctions
    #U100 = file.load_function(mesh, "Un", idx = 4)
    #u100, rho100, theta100, _ = U100.subfunctions
    u0 = exact_solution(x, y, 25.)
    U100 = file.load_function(mesh, "Un", idx = 1)
    file_test.write(u100)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    print(uerror)
    #thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    #thetaerrors[index] = thetaerror
    #print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    #print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    #print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index +=1


with CheckpointFile("checkpointVortex125.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =4)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    uerrors[index] = uerror
    thetaerrors[index] = thetaerror
    print("error u : ", sqrt(assemble(inner(u100 - u0,u100 - u0)*dx)))
    print("error rho : ", sqrt(assemble(inner(rho100 - rho0,rho100 - rho0)*dx)))
    print("error theta : ", sqrt(assemble(inner(theta100 - theta0, theta100 - theta0)*dx)))

index+=1


with CheckpointFile("checkpointVortex_time_0125.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =4)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    print(uerror, thetaerror)

with CheckpointFile("checkpointVortex_time_1125.h5", 'r') as file:
    mesh = file.load_mesh("mesh")
    print("meshes loaded")
    U0 = file.load_function(mesh, "Un", idx =0)
    u0, rho0, theta0, _ = U0.subfunctions
    U100 = file.load_function(mesh, "Un", idx =8)
    u100, rho100, theta100, _ = U100.subfunctions
    uerror = norm(u100 - u0)/norm(u0)
    thetaerror = norm(theta100 - theta0)/norm(theta0)
    print(uerror, thetaerror)
    

#this is run with dt=0.25
uerrors1 = np.array([2.39527350e-04, 1.03993314e-04, 2.52002043e-05, 8.81387842e-06,
 7.63333872e-06])
thetaerrors1 = np.array([0.02342123, 0.00712467, 0.00148584, 0.00048872, 0.00044297])
#run with dt = 0.25/4  
uerrors2 = np.array([2.39527350e-04, 1.03993314e-04, 2.52002043e-05, 8.81387842e-06,
 6.25414678e-06])
thetaerrors2 = np.array([2.34212273e-02, 7.12467366e-03, 1.48583516e-03, 4.88724639e-04,
 8.16547319e-05])
 # run with 0.25/8
uerrors3 = np.array([2.39527350e-04, 1.03993314e-04, 2.52002043e-05, 8.81387842e-06,
 6.24907077e-06])
thetaerrors3 = np.array([2.34212273e-02, 7.12467366e-03, 1.48583516e-03, 4.88724639e-04,
 7.76571841e-05])

mesh4 = np.array([1/10, 1/20, 1/40])
uerrors4 = np.array([0.00014115607259763824, 4.900111800791749e-05, 1.84659942398492e-05])

print(uerrors, thetaerrors) 
'''
hpower = mesh_sizes**2

uerrors10 = [4.494200796249309e-05,
0.00012539820224642258,
0.00020871120549821888,
0.00040062386282574284,
0.0004132880881321341,
0.00039080389176426183,
0.00025463725025615553,
0.00024022392133815244,
0.0002438636757042682
]
uerrors20 = [1.3845721697370717e-05,
4.278060571300691e-05,
0.00012276591910557617,
0.0004175922939542956,
0.00042847415113706026,
0.0004109123603119012,
0.0001459024017185981,
9.996744031881707e-05,
0.00010523615039596945
]
uerrors40 = [3.523530925931485e-06,
1.303160652190107e-05,
0.00010658445268578925,
0.0004276882391761728,
0.00043886409947050195,
0.00042611651944155627,
0.00011025891026478092,
2.4170309270076915e-05,
2.5395370403645503e-05
]
uerrors80 = [8.84805625751855e-07,
3.8107076052574546e-06,
0.00010477763376010187,
0.0004297581932764801,
0.0004411180765097968,
0.0004295102425859423,
0.00010563551943771601,
7.402305127954432e-06,
7.835295559280914e-06
]
uerrors10 =[]
uerrors20 =[]
uerrors40 =[]
uerrors80 =[]
uerrors125 =[]
uerrors160 =[]
with open("../Results/convergence/vortex10_uerrors.txt") as file:
    for line in file:
        uerrors10.append(float(line.rstrip()))
with open("../Results/convergence/vortex20_uerrors.txt") as file:
    for line in file:
        uerrors20.append(float(line.rstrip()))
with open("../Results/convergence/vortex40_uerrors.txt") as file:
    for line in file:
        uerrors40.append(float(line.rstrip()))
with open("../Results/convergence/vortex80_uerrors.txt") as file:
    for line in file:
        uerrors80.append(float(line.rstrip()))
with open("../Results/convergence/vortex125_uerrors.txt") as file:
    for line in file:
        uerrors125.append(float(line.rstrip()))

print(len(uerrors80))
i=-1
uerrors = np.array([uerrors10[i], uerrors20[i], uerrors40[i], uerrors80[i], uerrors125[i]])

rhoerrors10 =[]
rhoerrors20 =[]
rhoerrors40 =[]
rhoerrors80 =[]
rhoerrors125 =[]
rhoerrors160 =[]

with open("../Results/convergence/vortex10_rhoerrors.txt") as file:
    for line in file:
        rhoerrors10.append(float(line.rstrip()))
with open("../Results/convergence/vortex20_rhoerrors.txt") as file:
    for line in file:
        rhoerrors20.append(float(line.rstrip()))
with open("../Results/convergence/vortex40_rhoerrors.txt") as file:
    for line in file:
        rhoerrors40.append(float(line.rstrip()))
with open("../Results/convergence/vortex80_rhoerrors.txt") as file:
    for line in file:
        rhoerrors80.append(float(line.rstrip()))
with open("../Results/convergence/vortex125_rhoerrors.txt") as file:
    for line in file:
        rhoerrors160.append(float(line.rstrip()))

print(len(rhoerrors80))
rhoerrors = np.array([rhoerrors10[i], rhoerrors20[i], rhoerrors40[i], rhoerrors80[i], rhoerrors160[i]])

thetaerrors10 =[]
thetaerrors20 =[]
thetaerrors40 =[]
thetaerrors80 =[]
thetaerrors125 =[]
thetaerrors160 =[]

with open("../Results/convergence/vortex10_thetaerrors.txt") as file:
    for line in file:
        thetaerrors10.append(float(line.rstrip()))
with open("../Results/convergence/vortex20_thetaerrors.txt") as file:
    for line in file:
        thetaerrors20.append(float(line.rstrip()))
with open("../Results/convergence/vortex40_thetaerrors.txt") as file:
    for line in file:
        thetaerrors40.append(float(line.rstrip()))
with open("../Results/convergence/vortex80_thetaerrors.txt") as file:
    for line in file:
        thetaerrors80.append(float(line.rstrip()))
with open("../Results/convergence/vortex125_thetaerrors.txt") as file:
    for line in file:
        thetaerrors160.append(float(line.rstrip()))

print(len(rhoerrors80))
thetaerrors = np.array([thetaerrors10[i], thetaerrors20[i], thetaerrors40[i], thetaerrors80[i], thetaerrors160[i]])


fig, axes = plt.subplots()
axes.set_title("velocity errors")
plt.loglog(mesh_sizes, 0.8*1e-4*uerrors, color = "blue", label = "$e_{\mathrm{u}}$", marker = "x")
plt.loglog(mesh_sizes, hpower, color = "orange", label = "$h^2$")
axes.legend()
axes.set(xlabel='$ \log h$', ylabel='$ \log e_u$')
fig.savefig("uerrors.png")
plt.show()


fig, axes = plt.subplots()
axes.set_title("density errors")
plt.loglog(mesh_sizes, rhoerrors, color = "blue", label = "$e_{\mathrm{u}}$", marker = "x")
plt.loglog(mesh_sizes, hpower, color = "orange", label = "$h^2$")
axes.legend()
axes.set(xlabel='$\log h$', ylabel='$\log e_u$')
fig.savefig("rhoerrors.png")
plt.show()


fig, axes = plt.subplots()
axes.set_title("potential temperature errors")
plt.loglog(mesh_sizes, thetaerrors, color = "blue", label = "$e_{\mathrm{u}}$", marker = "x")
plt.loglog(mesh_sizes, hpower, color = "orange", label = "$h^2$")
axes.legend()
axes.set(xlabel='$ \log h$', ylabel='$ \log e_u$')
fig.savefig("thetaerrors.png")
plt.show()

'''
style = "classic"
with plt.style.context(style):
    fig, axes = plt.subplots()
    axes.set_title("Loglog plot of theta errors")
    plt.loglog(mesh_sizes, thetaerrors, color = "blue", label = "$e_{\mathrm{u}}$", marker = "x")
    plt.loglog(mesh_sizes, hpower, color = "orange", label = "h^2")
    axes.legend()
    plt.show()


available_styles = plt.style.available
n_styles = len(available_styles)

fig = plt.figure(dpi=100)
style = "classic"
with plt.style.context(style):

    ax = fig.add_subplot()
        
    plt.loglog(mesh_sizes, 1e-4*uerrors, color = "blue", label = "$e_u$", marker = "x")
    plt.loglog(mesh_sizes, hpower, color = "orange", label = "$h^2$", marker = ".")
    
    ax.tick_params(axis='x', labelrotation = 90)
    ax.set(xlabel='mesh size (normalised)', ylabel='$e_u$')
    ax.legend(loc='center right', ncol=4)
    ax.legend()
    ax.set_title("velocity errors")
    ax.set_xlim(left = 0.4*1e-2, right = 1.5*1e-1)
plt.show()
'''