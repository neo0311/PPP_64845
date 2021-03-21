import numpy as np
from material import *
import matplotlib.pyplot as plt

#E                          % Young's modulus
#v                          % Poisson's ratio
#sigmay0                    % initial yield stress
#H                          % kinematic hardening modulus
#h                          % isotropic hardening modulus
#G                          % shear modulus
#k                          % bulk modulus

def voigt_transform(X):
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
    """
    ################################Test
    C_voigt_values = []
    ###
    t_start = 0    
    t_end = Tf
    eps11_ = 0         #strain[1,1]
    sigma11_values = []  #sigma[1,1] values for plotting
    eps11_values = []   #strain[1,1] values for plotting
    delta_eps11 = load_max/(t_end/delta_t)
    num_time_steps = np.size(np.arange(t_start,t_end+delta_t, delta_t))

    #linear ramping of load
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
    for time in np.arange(t_start+delta_t,t_end+delta_t, delta_t):  #load stepping
        print('\n\n',step)
        if time==t_start:
            eps11 = 0
        elif t_start < time < t_end :
            
            eps11_ = eps11_ + delta_eps11
            eps11 = eps11_
        else:
            eps11 =load_max
        #print(time, round(eps11, 4))
        iteration = 0

        
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
            #print('\nC_out11', C_voigt[0,0], '\nC_out12', C_voigt[0,1], '\nC_out22', C_voigt[1,1], '\nC_out23', C_voigt[1,2], '\nC_out44', C_voigt[3,3])
            #print('E', E,'\nG', G, '\nk',k,'\nv', v,'\nsigmay0', sigmay0,'\nH', H,'\nh',h, '\neps', eps, '\nepspk', epsp_k,'\nAlpha_k', Alpha_k, '\nalpha_k', alpha_k,'\nepspk1' ,epsp_k1,'\nAlpha_k1', Alpha_k1, '\nalpha_k1', alpha_k1, '\nsigma', sigma, '\nC', voigt_transform(C) )
            #print('\neps', eps, '\nepspk', epsp_k,'\nAlpha_k', Alpha_k, '\nalpha_k', alpha_k,'\nepspk1' ,epsp_k1,'\nAlpha_k1', Alpha_k1, '\nalpha_k1', alpha_k1, '\nsigma', sigma, '\nC', C_voigt )
            print('eps11', eps[0,0],'eps22', eps[1,1], '\nepspk11', epsp_k[0,0],'\nepspk22', epsp_k[1,1],'\nAlpha_k11', Alpha_k[0,0],'\nAlpha_k22', Alpha_k[1,1], '\nalpha_k', alpha_k,'\nepspk_out11' ,epsp_k1[0,0],'\nepspk_out22' ,epsp_k1[1,1],'\nAlpha_k_out11', Alpha_k1[0,0], '\nAlpha_k_out22', Alpha_k1[1,1],'\nalpha_k_out', alpha_k1, '\nsigma_out11', sigma[0,0],'\nsigma_out22', sigma[1,1], '\nC_out11', C_voigt[0,0], '\nC_out12', C_voigt[0,1], '\nC_out22', C_voigt[1,1], '\nC_out23', C_voigt[1,2], '\nC_out44', C_voigt[3,3])
            
            ####dataset generation #################
            f = open('data_elasto_plastic.txt', 'a+')
            #print(5, file=f)
            print(eps[0,0],eps[1,1],epsp_k[0,0],epsp_k[1,1],Alpha_k[0,0],Alpha_k[1,1],alpha_k,epsp_k1[0,0],epsp_k1[1,1],Alpha_k1[0,0],Alpha_k1[1,1],alpha_k1,sigma[0,0],sigma[1,1],C_voigt[0,0],C_voigt[0,1],C_voigt[1,1],C_voigt[1,2],C_voigt[3,3],sep=',', file=f)
            f.close()
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
             
            #print(np.linalg.norm(sigma_bar))
            
            #stop condition
            if iteration == iterations or np.linalg.norm(sigma_bar)<tol:
                epsp_values[0,step], Alpha_values[0,step], alpha_values[0,step] = epsp_k, Alpha_k, alpha_k
                sigma11_values.append(sigma[0,0])
                eps11_values.append(eps11)
                break
        step+=1
        if step > max_step:
            break
    #fig, ax = plt.subplots()
    #ax.plot(eps11_values, sigma11_values)
    #ax.plot(eps11_values, C_voigt_values)

    #plt.show()
    
    #convert C to matrix form - remove rows and columns
    #convert stresses to voigt - remove first rows
    #same for strain

Tf = 10
delta_t = 1
load_max = 0.005
E = 210e+3   #Young's modulus
v = 0.33  #poison's ratio
sigmay0 = 200   #yield stress
H = 50*sigmay0
h = 10*sigmay0
G = E/(2*(1+v))
k = E/(3*(1-2*v))


####dataset preparation code#####
f= open('data_elasto_plastic.txt', 'w')
f.truncate(0)
f.write('eps11,eps22,epspk11,epspk22,Alpha_k11,Alpha_k22,alpha_k,epspk_out11,ep,epspk_out22,Alpha_k_out11,Alpha_k_out22,alpha_k_out,sigma_out11,sigma_out22,C_out11,C_out12,C_out22,C_out23,C_out44\n')
f.close()

for delta_t in (np.linspace(0.1,1,10)):
    print('delta_t',delta_t)
    global_routine(Tf, delta_t,load_max, E, G , k, v, sigmay0, H, h)

