from firedrake import ( ExtrudedMesh, FiniteElement, interval, HDiv, TensorProductElement,
    BrokenElement, EnrichedElement, WithMapping, FunctionSpace, quadrilateral)

'''
function to return (V0, Vv, V1, V2, T) for the 
velocity, vertical velocity, pressure, temperature. trace
in that order
'''

def build_spaces(mesh, vertical_degree, horizontal_degree):

    if vertical_degree is not None:
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
        V1 = FunctionSpace(mesh, V3_elt, name="DG") # pressure space
        V2 = FunctionSpace(mesh, V2t_elt, name="Temp")

        T = FunctionSpace(mesh, VT_elt, name = "Trace")

        remapped = WithMapping(V2v_elt_Broken, "identity") # only test with vertical part, drop Piola transformations

        Vv = FunctionSpace(mesh, remapped, name="Vv")

        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        DG1_space = FunctionSpace(mesh, DG1_elt, name = "DG1")

        #W_hydrostatic = MixedFunctionSpace((Vv, V1, T))

        # EDIT: return full spaces for full equations later

        return (V0, Vv, V1, V2, T)


def build_spaces_slice_3D(mesh, vertical_degree, horizontal_degree):

    ''' adds one cell in y direction to make Coriolis form implementable '''

    if vertical_degree is not None:
        # horizontal base spaces
        cell = mesh._base_mesh.ufl_cell().cellname()
        S1 = FiniteElement("RTCF", quadrilateral, horizontal_degree + 1)  # EDIT: family replaced by CG (was called with RT before)
        S2 = FiniteElement("DQ", quadrilateral, horizontal_degree, variant="equispaced")

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
        V1 = FunctionSpace(mesh, V3_elt, name="DG") # pressure space
        V2 = FunctionSpace(mesh, V2t_elt, name="Temp")

        T = FunctionSpace(mesh, VT_elt, name = "Trace")

        remapped = WithMapping(V2v_elt_Broken, "identity") # only test with vertical part, drop Piola transformations

        Vv = FunctionSpace(mesh, remapped, name="Vv")

        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        DG1_space = FunctionSpace(mesh, DG1_elt, name = "DG1")

        #W_hydrostatic = MixedFunctionSpace((Vv, V1, T))

        # EDIT: return full spaces for full equations later

        return (V0, Vv, V1, V2, T)