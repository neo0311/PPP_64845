import numpy as np

#Shape Functions
def N(zeeta):
    N = np.asarray(((1-zeeta)*(1/2), (1+zeeta)*(1/2)))
    N = np.reshape(N, (1,2))
    return N

#B matrix
def B(N, element_length, element_nodes):
    B = np.zeros((2,2))
    B[0,0] = -1/element_length
    B[0,1] = 1/element_length
    B[1,0] = N[0,0]/(N.dot(element_nodes))
    B[1,1] = N[0,1]/(N.dot(element_nodes))
    return B

#Determinant of Jacobian matrix
def J(element_length):
    return element_length/2

#Elemental tangent stiffness matrix
def tangent_stiffness_matrix(J, N, B, material_tangent_stiffness, element_nodes):
    Kte = 2* (B.T)@(material_tangent_stiffness)@(B) * N.dot(element_nodes) * J
    return Kte

#Elemental internal forces  
def f_internal(B, stress, N, J, element_nodes):
    f_int = 2* (B.T)@(stress) * N.dot(element_nodes) * J
    return f_int

#Elemental external force
def f_external(p, a, element):
    f_ext = np.zeros((2,1))
    if element == 1:
        f_ext[0,0] = p *a
    return f_ext