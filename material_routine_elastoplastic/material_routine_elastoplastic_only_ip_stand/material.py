import numpy as np


I2 = np.identity(3)  #2nd order identity tensor
I4 = np.einsum('ik,jl', I2,I2)  #4th order identity tensor
I4Sym = 1/2*(np.einsum('ik,jl', I2,I2) + np.einsum('il,jk', I2,I2))  #Symmetric part of 4th order identity tensor
P4Sym = I4Sym - 1/3*np.einsum('ij,kl', I2,I2)    #The deviatorisor tensor

def frobenius_norm(X):
    """
    Calculates frobenius norm
    """
    return np.sqrt(np.einsum('ij,ij', X,X))

def material(E, G, k,v, sigmay0, H,h, eps, epsp, Alpha, alpha):
    """
    Material routine for small strain plasticity with linear isotropic and kinematic hardening
    """
    tolerance = 1e-8
    #dev_eps = np.zeros()
    dev_eps = eps - 1/3 * np.trace(eps)*I2   #deviatoric part of total strain
    dev_sigma_tr = 2 * G * (dev_eps - epsp)  #devaitoric stress trial
    B_tr = H * Alpha  #trial value of tensorial internal state variable 
    b_tr = h *alpha  #trial value of scalar internal state variable
    xi_tr = dev_sigma_tr - B_tr  #trial value of stress difference vector
    xi_tr_norm = frobenius_norm(xi_tr) #norm of xi_tr
    n_tr = xi_tr/xi_tr_norm  #plastic flow direction
    phi_tr = xi_tr_norm - np.sqrt(2/3)*(sigmay0 + b_tr)  #trial value of yield function

    #check for yielding
    if phi_tr < tolerance:   
        #using old values for no yielding
        dev_sigma = dev_sigma_tr
        dev_C = 2 * G * P4Sym

    else:
        #plastic correction

        pl_multi = phi_tr/(2*G + H + (2/3)*h)   #plastic multiplier
        
        #new values of various variables
        dev_sigma = dev_sigma_tr - 2*G*pl_multi*n_tr
        epsp = epsp + (dev_sigma_tr-dev_sigma)/(2*G)
        Alpha = Alpha + pl_multi*n_tr
        alpha = alpha + np.sqrt(2/3)*pl_multi
        beta1 = 1 - (phi_tr/xi_tr_norm)*(1/(1 + (H/(2*G)) + h/(3*G)))
        beta2 = (1 - phi_tr/xi_tr_norm)*(1/(1 + (H/(2*G)) + h/(3*G)))
        dev_C = 2*G*beta1*P4Sym - 2*G*beta2*(np.einsum('ij,kl',n_tr, n_tr))  # deviatoric part of elastic stiffness tensor
    
    #calculating values of stresses and algorithmic tangent stiffness matrix for the current time step
    sigma = dev_sigma + k*np.trace(eps)*I2
    C = dev_C + k*np.einsum('ij,kl', I2,I2)

    return epsp, Alpha, alpha, sigma, C

    

