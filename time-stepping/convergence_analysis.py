
import matplotlib.pyplot as plt
import numpy as np
from firedrake import *

'''
Lauter test, Example 3 in https://doi.org/10.1016/j.jcp.2005.04.022
with dt = 400.

from file unsteady_solid_body rotation 
'''
u_errors3 = []
D_errors3 = []

u_errors4 = []
D_errors4 = []

u_errors5 = []
D_errors5 = []

u_errors6 = []
D_errors6 = []
uref = 712008703.4132433 
dref = 298004433314.0549

refinement_levels = [3,4,5,6]
refinement_level=refinement_levels[0]
'''
R0 = 6371220.
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = IcosahedralSphereMesh(radius=R0,
                            degree=1,
                            refinement_level=refinement_level,
                            distribution_parameters = distribution_parameters)
print("Mesh shize level 4 = ", mesh.cell_sizes())
'''
mesh_sizes = [1, 1/2, 1/4, 1/8]
path = "Results/convergence-errors/"

with open(path+"uerrors3.txt", 'r') as ufile:
    for item in ufile:
        u_errors3.append(float(item))
with open(path +"Derrors3.txt", 'r') as Dfile:
    for item in Dfile:
        D_errors3.append(float(item))

with open(path+"uerrors4.txt", 'r') as ufile:
    for item in ufile:
        u_errors4.append(float(item))
with open(path+"Derrors4.txt", 'r') as Dfile:   
    for item in Dfile:  
        D_errors4.append(float(item))

with open(path+"uerrors5.txt", 'r') as ufile:
    for item in ufile:
        u_errors5.append(float(item))
with open(path+"Derrors5.txt", 'r') as Dfile:   
    for item in Dfile:  
        D_errors5.append(float(item))

with open(path+"uerrors6.txt", 'r') as ufile:
    for item in ufile:
        u_errors6.append(float(item))
with open(path+"Derrors6.txt", 'r') as Dfile:   
    for item in Dfile:  
        D_errors6.append(float(item))
'''
times = 400*100*np.arange(len(u_errors4))

fig, axs = plt.subplots(2)
print(len(u_errors3))

axs[0].plot(times,u_errors3, label='u errors level 3')
axs[1].plot(times,D_errors3, label='D errors level 3')
axs[0].plot(times,u_errors4, label='u errors level 4')
axs[1].plot(times,D_errors4, label='D errors level 4')
axs[0].plot(times,u_errors5, label='u errors level 5')
axs[1].plot(times,D_errors5, label='D errors level 5')
axs[0].plot(times,u_errors6, label='u errors level 6')
axs[1].plot(times,D_errors6, label='D errors level 6')

lines_labels = axs[0].get_legend_handles_labels()
axs[0].legend(*lines_labels, loc='upper center', ncol=4)
lines_labels2 = axs[1].get_legend_handles_labels()
axs[1].legend(*lines_labels2, loc='upper center', ncol=4)

plt.show()
'''
print(len(u_errors3))
print(len(u_errors6))

u_errors3 = np.array(u_errors3)/uref
u_errors4 = np.array(u_errors4)/uref
u_errors5 = np.array(u_errors5)/uref
u_errors6 = np.array(u_errors6)/uref

D_errors3 = np.array(D_errors3)/dref
D_errors4 = np.array(D_errors4)/dref
D_errors5 = np.array(D_errors5)/dref
D_errors6 = np.array(D_errors6)/dref

h = 2.0**(-np.array(refinement_levels))
fig, axs = plt.subplots(2)
T=17
plt.suptitle("$L^2$-errors after T = 1d")
axs[0].loglog(h, [u_errors3[T], u_errors4[T], u_errors5[T], u_errors6[2*T]], marker = 'x')
axs[0].loglog(h, 1e-1*h**2)
axs[0].legend(["$e_{\mathrm{u}}$", "$h^2$"])

axs[1].loglog(h,[D_errors3[T], D_errors4[T], D_errors5[T], D_errors6[2*T]], marker = 'x')
axs[1].loglog(h, 1e-2*h**2)
plt.legend(["$e_D$", "$h^2$"])
axs[0].set(xlabel='$\log h$', ylabel='$\log e_u$')
axs[1].set(xlabel='$\log h$', ylabel='$\log e_D$')

#plt.show()
plt.savefig("convergence-plot.png", dpi=1000)
plt.show()

print(len(u_errors6))