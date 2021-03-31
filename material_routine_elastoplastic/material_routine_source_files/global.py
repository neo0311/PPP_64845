import numpy as np
from material import *
import matplotlib.pyplot as plt

###Consistant driver algorithm for checking the validity of material routine outside FEM framework.###
###Coded by myself under PPP##
###Used for generating data###


def voigt_transform(X):
    """
    Function to transform normal representation to  voigt representation.
    X     : nd tensor(stresses, strains or elastic stiffness tensor)
    Voigt : respective voigt representations
    """

    if np.array_equal(np.shape(X), np.asarray((3,3,3,3)))==True:
        """
        Transforms the given stress, strain and elasticity tensor to voigt matrices
        source: https://github.com/libAtoms/matscipy
        """
        tol = 1e-3
        Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)] 
        C = np.asarray(X)
        Voigt = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                k, l = Voigt_notation[i]
                m, n = Voigt_notation[j]
                Voigt[i,j] = C[k,l,m,n]

    elif np.array_equal(np.shape(X), np.asarray([3,3]))==True: #converts stress and strain tensors to voigt arrays
        Voigt= np.zeros((6,1))
        Voigt[0,0] = X[0,0]
        Voigt[1,0] = X[1,1]
        Voigt[2,0] = X[2,2]
        Voigt[3,0] = X[1,2]
        Voigt[4,0] = X[0,2]
        Voigt[5,0] = X[0,1]
    
    else:
        raise ValueError("Invalid")

    return Voigt

def voigt_reverse_transform(X_voigt):
    """
    Transforms voigt arrays to respective tensors
    X_voigt  : array in voigt representation.
    """
    Voigt  = np.zeros((3,3))
    if np.array_equal(np.shape(X_voigt), np.asarray([6,1])) and np.size(X_voigt)==6:
        Voigt[0,0] = X_voigt[0,0]
        Voigt[1,1] = X_voigt[1,0]
        Voigt[2,2] = X_voigt[2,0]
        Voigt[1,2], Voigt[2,1] = X_voigt[3,0], X_voigt[3,0]
        Voigt[0,2], Voigt[2,0] = X_voigt[4,0], X_voigt[4,0]
        Voigt[0,1], Voigt[1,0] = X_voigt[5,0], X_voigt[5,0]
    return Voigt

def global_routine(Tf, delta_t,load_max, E=0, G =0, k=0, v=0, sigmay0=0, H=0, h=0):
    """
    Driver routine to check the respective material routine
        E                          :Young's modulus
        v                          :Poisson's ratio
        sigmay0                    :initial yield stress
        H                          :kinematic hardening modulus
        h                          :isotropic hardening modulus
        G                          :shear modulus
        k                          :bulk modulus
        Tf                         :Final loading time
        delta_t                    :time interval for loading
        load_max                   :maximum loading
    """
    
    
    ################################Test
    C_voigt_values = []
    ###
    
    #initialisation of variables
    t_start = 0    
    t_end = Tf
    eps11_ = 0         #strain[1,1]
    sigma11_values = []  #sigma[1,1] values for plotting
    eps11_values = []   #strain[1,1] values for plotting
    delta_eps11 = load_max/(t_end/delta_t)
    num_time_steps = np.size(np.arange(t_start,t_end+delta_t, delta_t))
    eps = np.zeros((3,3))  
    eps_bar = np.zeros((5,1))                                 #current strain matrix
    eps_values = np.zeros((1,num_time_steps), dtype=object)     #array for storing strain matrices
    epsp = np.zeros((3,3))
    epsp_values = np.zeros((1,num_time_steps), dtype=object)
    epsp_values[0,0] = epsp
    Alpha_values = np.zeros((1,num_time_steps), dtype=object)
    Alpha = np.zeros((3,3))
    Alpha_values[0,0] = Alpha
    alpha_values = np.zeros((1,num_time_steps))
    tol = 1e-10
    iterations=100
    step = 1
    
    max_step =150
    #linear ramping of load

    for time in np.arange(t_start+delta_t,t_end+delta_t, delta_t):  #load stepping
        print('step:',step)
        print('time:',time)
        if time==t_start:
            eps11 = 0
        elif t_start < time < t_end :
            
            eps11_ = eps11_ + delta_eps11
            eps11 = eps11_
        else:
            eps11 =load_max
        #print(time, round(eps11, 4))
        iteration = 0

        print('load', eps11)
        sigma_bar = np.ones((5,1))
        delta_sigma_bar = np.zeros((5,1))
        C_bar = np.zeros((5,5))
        
        #evolution variables initiation
        epsp_k = epsp_values[0,step-1] 
        Alpha_k = Alpha_values[0,step-1] 
        alpha_k = alpha_values[0,step-1]

        while True:
            """
            Newton rapson loop for solving the non linear system of equations
            """
            iteration +=1
            eps[0,0] = eps11
            eps_values[0,step] = eps

            #calling material routine
            epsp_k1, Alpha_k1, alpha_k1, sigma, C = material(E, G, k,v, sigmay0, H,h, eps, epsp_k, Alpha_k, alpha_k)
            C_voigt = voigt_transform(C)
            
            epsp_k, Alpha_k, alpha_k = epsp_k1, Alpha_k1, alpha_k1

            #reduction
            sigma_bar = np.delete(voigt_transform(sigma), 0, axis=0)
            C_bar = voigt_transform(C)[1:,1:]
            
            #solving the equation for strain increments
            delta_sigma_bar = -1 *(np.linalg.inv(C_bar) @ sigma_bar)
            eps_bar = eps_bar + delta_sigma_bar
            
            #updating strain tensors with new values
            eps_voigt = voigt_transform(eps)
            eps_voigt[1:,:] = eps_bar
            eps = voigt_reverse_transform(eps_voigt)
                         
            #stop condition
            if iteration == iterations or np.linalg.norm(sigma_bar)<tol:
                epsp_values[0,step], Alpha_values[0,step], alpha_values[0,step] = epsp_k, Alpha_k, alpha_k
                sigma11_values.append(sigma[0,0])
                eps11_values.append(eps11)
                print('stress:\n', sigma, '\n')
                break
        step+=1
        if step > max_step:
            break
    fig, ax = plt.subplots()
    ax.plot(eps11_values, sigma11_values)
    plt.xlabel('strain')
    plt.ylabel('stress')
    plt.grid()
    plt.title('Stress strain plot')
    plt.savefig('stress_strain_plot.png')
    plt.show()
    

#material paramters used##
Tf = 10
delta_t = 0.001
load_max = 0.5
E = 210e+3   #Young's modulus
v = 0.33  #poison's ratio
sigmay0 = 200   #yield stress
H = 50*sigmay0
h = 10*sigmay0
G = E/(2*(1+v))
k = E/(3*(1-2*v))


#executing the function.
global_routine(Tf, delta_t,load_max, E, G , k, v, sigmay0, H, h)

