import numpy as np


def mesh(inner_radius,outer_radius, number_of_elements, mesh_refinement_factor):
    """Returns the nodal positions"""
    if number_of_elements > 1:
        q = mesh_refinement_factor ** (1/(number_of_elements-1))
        dr=(outer_radius-inner_radius)*(1-q)/(1-mesh_refinement_factor*q)
        rnode = inner_radius
        rnodes = []
        rnodes.append(inner_radius)
        for i in range(number_of_elements):
            rnode = rnode + dr
            rnodes.append(rnode)
            dr = q*dr
        return np.asarray(rnodes)

    else:
        return np.asarray((inner_radius, outer_radius))

