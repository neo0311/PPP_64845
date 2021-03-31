import numpy as np


#Elastic stiffness matrix
def C(E, v):
    """Elastic stiffness matrix"""
    C = np.zeros((2,2))
    C[0,0] = (E/((1+v)*(1-2*v)))*(1-v)
    C[0,1] = (E/((1+v)*(1-2*v)))*v
    C[1,0] = (E/((1+v)*(1-2*v)))*v
    C[1,1] = (E/((1+v)*(1-2*v)))*(1-v)
    return C

#strain
def strain(B,u):
    return B@u

#over stress
def over_stress(Q, over_stress_old, delta_strain, tL, tF, delta_t, T):
    over_stress_new = np.zeros((2,1))
    dev_delta_strain = delta_strain - (1/3)*np.reshape((np.sum(delta_strain),np.sum(delta_strain)), (2,1))
    over_stress_new = (1/(1+(delta_t/(2*T)))) * (over_stress_old * (1 - (delta_t/(2*T))) + Q * dev_delta_strain)
    over_stress = over_stress_new
    return over_stress

#total stress
def stress(C, strain, over_stress):
    return C@(strain) + over_stress

#material_tangent_stiffness matrix
def material_tangent_stiffness(C, delta_t, Q, T):
    strain_derivative = np.ones((2,2))*(-1/3) + np.identity(2)
    Ct = C + (Q/(1+(delta_t/(2*T))))*strain_derivative
    return Ct
