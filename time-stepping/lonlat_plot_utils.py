from firedrake import *
from galewsky_utils import *

R0 = 6371220.

def lonlatr_from_xyz(x, y, z, angle_units='rad'):
    """
    Returns the spherical lon, lat and r coordinates from the global geocentric
    Cartesian x, y, z coordinates.

    Args:
        x (:class:`np.ndarray` or :class:`ufl.Expr`): x-coordinate.
        y (:class:`np.ndarray` or :class:`ufl.Expr`): y-coordinate.
        z (:class:`np.ndarray` or :class:`ufl.Expr`): z-coordinate.
        angle_units (str, optional): the units to use for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (lon, lat, r) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    # Determine whether to use firedrake or numpy functions
    #module, _ = firedrake_or_numpy(x)
    #atan2 = module.atan2 if hasattr(module, "atan2") else module.arctan2
    #sqrt = module.sqrt
    #pi = module.pi

    if angle_units == 'deg':
        unit_factor = 180./pi
    if angle_units == 'rad':
        unit_factor = 1.0

    lon = atan2(y, x)
    r = sqrt(x**2 + y**2 + z**2)
    l = sqrt(x**2 + y**2)
    lat = atan2(z, l)

    return lon*unit_factor, lat*unit_factor, r

def xyz_from_lonlatr(lon, lat, r, angle_units='rad'):
    """
    Returns the geocentric Cartesian coordinates x, y, z from spherical lon, lat
    and r coordinates.

    Args:
        lon (:class:`np.ndarray` or :class:`ufl.Expr`): longitude coordinate.
        lat (:class:`np.ndarray` or :class:`ufl.Expr`): latitude coordinate.
        r (:class:`np.ndarray` or :class:`ufl.Expr`): radial coordinate.
        angle_units (str, optional): the units used for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (x, y, z) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    # Import routines
    #module, _ = firedrake_or_numpy(lon)
    #cos = module.cos
    #sin = module.sin
    #pi = module.pi

    if angle_units == 'deg':
        unit_factor = pi/180.0
    if angle_units == 'rad':
        unit_factor = 1.0

    lon = lon*unit_factor
    lat = lat*unit_factor

    x = r * cos(lon) * cos(lat)
    y = r * sin(lon) * cos(lat)
    z = r * sin(lat)

    return x, y, z



def get_flat_latlon_mesh(mesh):
    """
    Function from gusto -- plot looks distorted
    Construct a planar latitude-longitude mesh from a spherical mesh.

    Args:
        mesh (:class:`Mesh`): the mesh on which the simulation is performed.
    """
    coords_orig = mesh.coordinates
    coords_fs = coords_orig.function_space()

    if coords_fs.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    coords_dg = Function(vec_DG1).interpolate(coords_orig)
    coords_latlon = Function(vec_DG1)
    shapes = {"nDOFs": vec_DG1.finat_element.space_dimension(), 'dim': 3}

    radius = np.min(np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:, 0] = np.arctan2(coords_dg.dat.data[:, 1], coords_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:, 1] = np.arcsin(coords_dg.dat.data[:, 2]/np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # our vertical coordinate is radius - the minimum radius
    coords_latlon.dat.data[:, 2] = np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2) - radius

# We need to ensure that all points in a cell are on the same side of the branch cut in longitude coords
# This kernel amends the longitude coords so that all longitudes in one cell are close together
    kernel = op2.Kernel("""
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
void splat_coords(double *coords) {{
    double max_diff = 0.0;
    double diff = 0.0;

    for (int i=0; i<{nDOFs}; i++) {{
        for (int j=0; j<{nDOFs}; j++) {{
            diff = coords[i*{dim}] - coords[j*{dim}];
            if (fabs(diff) > max_diff) {{
                max_diff = diff;
            }}
        }}
    }}

    if (max_diff > PI) {{
        for (int i=0; i<{nDOFs}; i++) {{
            if (coords[i*{dim}] < 0) {{
                coords[i*{dim}] += TWO_PI;
            }}
        }}
    }}
}}
""".format(**shapes), "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return Mesh(coords_latlon)

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

'''
mesh_sphere = IcosahedralSphereMesh(radius=R0,
                            degree=1,
                            refinement_level=5,
                            distribution_parameters = distribution_parameters)
x = SpatialCoordinate(mesh_sphere)
mesh_sphere.init_cell_orientations(x)
Dexpr = depth_expression(*x)
u_expr = velocity_expression(*x)
V = FunctionSpace(mesh_sphere, "BDM", degree=2)
V2 = FunctionSpace(mesh_sphere, "DG", degree=1)
u = Function(V).project(u_expr)
e1 = as_vector([1,0,0])
e2 = as_vector([0,1,0])
e3 = as_vector([0,0,1])
u_mag = dot(u,e1)**2 + dot(u,e2)**2+ dot(u,e3)**2

u_mag = Function(V2).interpolate(u_mag)
D = Function(V2).interpolate(Dexpr)
'''
'''
mesh_ll = get_flat_latlon_mesh(mesh)
x = SpatialCoordinate(mesh_ll)
mesh_ll.init_cell_orientations(x)
field = Function(functionspaceimpl.WithGeometry.create(
                        u.function_space(), mesh_ll),
                    val=u.topological, name='mesh_ll')

file = File("test_latlong.pvd")
file.write(field)
'''


'''
Try David's version
X 1. Make a rectangle mesh using lat-lon coordinates.
X 2. create a CG1 (or higher) 3D VectorFunctionSpace on the same mesh.
X 3. Interpolate the change of coordinates function onto a Function in the 3D space.
X 4. Pass that function to the Mesh constructor. This gives you a lat-lon mesh with the same degree of freedom layout as the rectangle mesh.
5. Interpolate your solution from the sphere mesh onto your 3D lat-lon mesh (this works using the cross-mesh interpolation we now have).
6. Copy the dats from the 3D lat-lon Function into the dats of a corresponding Function on the 2D mesh.
7. Profit!
'''


file = File("test_latlong.pvd")

def sphere_to_latlongrect(f):
    mesh_rect = RectangleMesh(nx = 200, ny=200, Lx = pi, Ly = pi/2-1/1000000, originX = -pi, originY = -pi/2+1/1000000)
    CG1 = VectorFunctionSpace(mesh_rect,"CG", degree =2, dim=3)
    lon, lat = SpatialCoordinate(mesh_rect)

    cx,cy,cz = xyz_from_lonlatr(lon,lat, r=R0)
    coord_interpolated = Function(CG1).interpolate(as_vector([cx,cy,cz]))
    mesh = Mesh(coord_interpolated)

    #Dexpr = depth_expression(*x)
    V3 = FunctionSpace(mesh, "CG", degree =1)
    f_3d = Function(V3).interpolate(f)

    V2 = FunctionSpace(mesh_rect, "CG", degree=1)
    f_ll = Function(V2)
    f_ll.dat.data_wo[:] = f_3d.dat.data_ro[:]
    return f_ll
    
mesh_rect = RectangleMesh(nx = 200, ny=200, Lx = pi, Ly = pi/2-1/1000000, originX = -pi, originY = -pi/2+1/1000000)
CG = VectorFunctionSpace(mesh_rect,"CG", degree =3, dim=3)
lon, lat = SpatialCoordinate(mesh_rect)

cx,cy,cz = xyz_from_lonlatr(lon,lat, r=R0)
coord_interpolated = Function(CG).interpolate(as_vector([cx,cy,cz]))
mesh_ll = Mesh(coord_interpolated)
V_ll_sphere = FunctionSpace(mesh_ll, "CG", degree =2) #to project function onto this 3d sphere ll-mesh

def get_V_ll_rect():
    return FunctionSpace(mesh_rect, "CG", degree=2)

V_rect = get_V_ll_rect # create functio on 2d rectangular mesh and copy data into this

e1 = as_vector([1,0,0])
e2 = as_vector([0,1,0])
e3 = as_vector([0,0,1])


def sphere_to_latlongrect(f, f_ll):
    '''
        f_ll expected to be function on rectangular lonlat mesh
        result written into f_ll
    '''
    f_3d = Function(V_ll_sphere).interpolate(f)
    f_ll.dat.data_wo[:] = f_3d.dat.data_ro[:]

def sphere_to_latlongrect_vec(VS, f, f_ll):
    '''
        when f is a vector 
        VS: function space on original mesh (usually Icosahedralspheremesh)
        
    '''
    f_mag = Function(VS).interpolate(dot(f,e1)**2 + dot(f,e2)**2+ dot(f,e3)**2)
    f_3d = Function(V_ll_sphere).interpolate(f_mag)
    f_ll.dat.data_wo[:] = f_3d.dat.data_ro[:]
