from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

mesh_sizes = np.array([1/10, 1/20, 1/40, 1/80, 1/160]) 
hpower = mesh_sizes**2
nx_list = [10, 20, 40, 80, 160]
uerrors = np.zeros(len(mesh_sizes))
thetaerrors = np.zeros(len(mesh_sizes))

uc = 100.
vc = 100.
L = 10000.
H = 10000.

uerrors10 =[]
uerrors20 =[]
uerrors40 =[]
uerrors80 =[]
uerrors125 =[]
uerrors160 =[]

with open("test_adv10_uerrors.txt") as file:
    for line in file:
        uerrors10.append(float(line.rstrip()))
with open("test_adv20_uerrors.txt") as file:
    for line in file:
        uerrors20.append(float(line.rstrip()))
with open("test_adv40_uerrors.txt") as file:
    for line in file:
        uerrors40.append(float(line.rstrip()))
with open("test_adv80_uerrors.txt") as file:
    for line in file:
        uerrors80.append(float(line.rstrip()))
with open("test_adv160_uerrors.txt") as file:
    for line in file:
        uerrors160.append(float(line.rstrip()))
i=-1
print(len(uerrors10))

uerrors = np.array([uerrors10[i], uerrors20[i], uerrors40[i], uerrors80[i], uerrors160[i]])
print(uerrors)

fig, axes = plt.subplots()
axes.set_title("Loglog plot of velocity errors")
plt.loglog(mesh_sizes, uerrors, color = "red", label = "u-error", marker = ".")
plt.loglog(mesh_sizes, 1e4*hpower, color = "orange", label = "h^2", marker = ".")
axes.legend()
fig.savefig("uerrors.png")
plt.show()
