import numpy as np
I2 = np.identity(3)
#I4 = np.tensordot(np.identity(3), np.identity(3), axes=0)
I4 = np.einsum('ik,jl', I2,I2)
#I2Sym = 1/2*(I2 + I2.T)
I4Sym = 1/2*(np.einsum('ik,jl', I2,I2) + np.einsum('il,jk', I2,I2))
P4Sym = I4Sym - 1/3*np.einsum('ij,kl', I2,I2)
#I4Skw = 1/2*(np.einsum('ik,jl', I2,I2) - np.einsum('il,jk', I2,I2))

#print(1/2*(np.einsum('ik,jl', I2,I2) + np.einsum('il,jk', I2,I2)))
def frobenius_norm(X):
    return np.sqrt(np.einsum('ij,ij', X,X))

def material(E, G, k,v, sigmay0, H,h, eps, epsp, Alpha, alpha):

    #print(E, G, k,v, sigmay0, H,h, eps, epsp, Alpha, alpha)
    tolerance = 1e-8
    #dev_eps = np.zeros()
    dev_eps = eps - 1/3 * np.trace(eps)*I2
    dev_sigma_tr = 2 * G * (dev_eps - epsp)
    B_tr = H * Alpha
    b_tr = h *alpha
    xi_tr = dev_sigma_tr - B_tr
    xi_tr_norm = frobenius_norm(xi_tr)
    n_tr = xi_tr/xi_tr_norm
    phi_tr = xi_tr_norm - np.sqrt(2/3)*(sigmay0 + b_tr)

    if phi_tr < tolerance:
        dev_sigma = dev_sigma_tr
        dev_C = 2 * G * P4Sym

    else:
        pl_multi = phi_tr/(2*G + H + (2/3)*h)
        dev_sigma = dev_sigma_tr - 2*G*pl_multi*n_tr
        epsp = epsp + (dev_sigma_tr-dev_sigma)/(2*G)
        Alpha = Alpha + pl_multi*n_tr
        alpha = alpha + np.sqrt(2/3)*pl_multi
        beta1 = 1 - (phi_tr/xi_tr_norm)*(1/(1 + (H/(2*G)) + h/(3*G)))
        beta2 = (1 - phi_tr/xi_tr_norm)*(1/(1 + (H/(2*G)) + h/(3*G)))
        dev_C = 2*G*beta1*P4Sym - 2*G*beta2*(np.einsum('ij,kl',n_tr, n_tr))
    
    sigma = dev_sigma + k*np.trace(eps)*I2
    C = dev_C + k*np.einsum('ij,kl', I2,I2)

    return epsp, Alpha, alpha, sigma, C

    

