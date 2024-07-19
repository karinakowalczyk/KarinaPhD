from firedrake import *
from TimeSteppingClass import *
import timeit
import matplotlib.pyplot as plt

'''
Lauter test, Example 3 in https://doi.org/10.1016/j.jcp.2005.04.022
'''

R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
Omega_vec = Omega*as_vector([0.,0.,1.])
g = Constant(9.8)  # Gravitational constant

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
refinement_level_list= [3,4,5,6]
n_ref = len(refinement_level_list)

uerror = []
Derror = []

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

    e1 = as_vector([1.,0.,0.])
    e2 = as_vector([0.,1.,0.])
    e3 = as_vector([0.,0.,1.])
    b1 = cos(Omega*t)*e1 + sin(Omega*t)*e2
    b2 = -sin(Omega*t)*e1 + cos(Omega*t)*e2
    b3 = e3


    alpha = pi/4
    c_vec = -sin(alpha)*e1 + cos(alpha)*e3

    #phi = dot(c_vec, b1)*e1 +dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3 
    #phi_dot_n = dot(phi, n)

    u0 = (2*pi*R0/12)/86400
    #u0 = 20.
    H= 133681./g
    
    def exact_sol(t):
        b1 = cos(Omega*t)*e1 + sin(Omega*t)*e2
        b2 = -sin(Omega*t)*e1 + cos(Omega*t)*e2
        b3 = e3
        r = (x[0]**2+x[1]**2+x[2]**2)**0.5
        k = as_vector(x)/r
        c_vec = -sin(alpha)*e1 + cos(alpha)*e3
        phi_t = dot(c_vec, b1)*e1 +dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3 
        phi_t_dot_n = dot(phi_t, k)
        u_exact = u0 * cross(phi_t, k)
        z = x[2]/r*R0
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

    uerror.append(norm(u_ex - u_ex_0))
    Derror.append(1.e-9*norm(D_ex - D_ex_0))

print(Derror)
h = 2.0**(-np.array(refinement_level_list))
plt.loglog(h, np.array(Derror))
plt.loglog(h, h**2)
plt.legend(["D error", "$h^2$"])

#plt.loglog(Derror)

plt.show()
'''
htest = 2.0**(-np.array([0,1,2,3]))
print(htest)
utest = np.array(1e8*htest**2)
print(utest)
plt.loglog(htest,utest)
plt.loglog(htest, 1e8*htest**2)
plt.legend(["test error", "$h^2$"])
plt.show()
'''