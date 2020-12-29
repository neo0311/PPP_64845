import numpy as np
import element_routine
from element_routine import *
from material_routine import *
from mesh import *
import matplotlib.pyplot as plt
import time as ti
  
###Given parameters
E = 100000
v = 0.30
Q = 50000
T = 2
a = 40
b = 80
p_max = 70
tL = 4
tF = 20
mesh_refinement_factor = 2


def main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements):
    

    zeeta = 0
    rnodes = mesh(a,b, num_of_elements, mesh_refinement_factor)
    num_of_iterations = np.size(np.arange(0,tF+delta_t, delta_t))

    def assignment_matrix(i, num_elements):
        """Transformation matrix calculation"""
        A = np.zeros((2,num_elements+1))
        A[0,0] = 1
        A[1,1] = 1
        A[0] = np.roll(A[0], i-1)
        A[1] = np.roll(A[1], i-1)
        return A

    ##analytical solution
    def analytical_elastic(E, v,p,a,b, element_nodes):
        """Analytical solution"""
        r = element_nodes
        return (1+v) * (p/E) * (a**2/(b**2 - a**2)) * ((1-2*v)*r + (b**2/r))

    def Infinity_norm(X):
        """returns Infinity Norm"""
        return np.linalg.norm(X, np.inf)

    
    ##load scaling
    delta_p = p_max/(tL/delta_t)
    ps = 0
    p = 0

    ##create an empty file
    
    ue = np.zeros((2,1))
    um = np.zeros((num_of_elements+1,1))
    uk = np.zeros((num_of_elements+1, 1))
    uvalues = np.zeros((num_of_elements+1, num_of_iterations))
    strain_values = np.zeros((num_of_elements+1, num_of_iterations), dtype=object)
    over_stress_values = np.zeros((num_of_elements+1, num_of_iterations), dtype=object)
    stress_values = np.zeros((num_of_elements+1, 2))
    m = 1
    delta_strain = np.zeros((2,1))
    count = 1
    for time in np.arange(delta_t,tF+delta_t,delta_t):

        """Looping in time to find the displacements"""
        ##load scaling
        if time < tL :
            ps = ps + delta_p
            p = ps
        else:
            p =p_max
        uk = um
        nr = 0  ##NR runs counter
        delta_uk = np.zeros((num_of_elements+1,1))
        um = um*0
        
        while True:

            """Newton Rapson for finding the displacement um at the given time"""
            Kt = np.zeros((num_of_elements + 1,num_of_elements+1))
            Rk = np.zeros((num_of_elements+1, 1))
            F_ext = np.zeros((num_of_elements+1, 1))
            F_int = np.zeros((num_of_elements+1, 1))
            
            for e in range(1,num_of_elements+1):  ###assembling of internal and external forces and calculating displacements

                ##elemental structural data calculation
                element_length = rnodes[e] - rnodes[e-1]
                element_nodes = np.reshape(np.asanyarray((rnodes[e-1], rnodes[e])),(2,1))
                A = assignment_matrix(e, num_of_elements)
                ue = A@uk

                ##elemental material data calculation

                C_ = C(E, v)
                N_ = N(0)
                J_ = J(element_length)
                B_ = B(N_, element_length, element_nodes)
                Ct = material_tangent_stiffness(C_, delta_t, Q, T)
                Kte = tangent_stiffness_matrix(J_, N_, B_, Ct, element_nodes)
                strain_values[e,m] = strain(B_,ue)
                delta_strain = strain_values[e,m]- strain_values[e,m-1]
                over_stress_values[e,m]= over_stress(Q,over_stress_values[e,m-1], delta_strain, tL, tF, delta_t, T)
                stress_ = stress(C_, strain_values[e,m], over_stress_values[e,m])
                stress_values[e,0] = stress(C_, strain_values[e,m], over_stress_values[e,m])[0]  ##saving sigma rr for result extraction
                stress_values[e,1] = stress(C_, strain_values[e,m], over_stress_values[e,m])[1]  ##saving sigma phi for result extraction

                ##Elemental forces
                f_internal_e = f_internal(B_, stress_, N_, J_ ,element_nodes)
                f_external_e = f_external(p,a,e)
                #element_end_time = ti.time()
                #print('element_duration', (element_end_time - element_start_time), e)
                Kt = Kt + ((A.T)@(Kte)@(A))
                Rk = Rk + A.T @ (f_internal_e- f_external_e)
                ##Total Forces
                F_ext = F_ext + A.T@(f_external_e)
                F_int = F_int + A.T.dot(f_internal_e)
                #elaspsed_time= start_time - ti.time()
                f = open('data_linear_elastic.txt', 'a+')
                #print(E,v,element_length,Q,T,p,f_internal_e[0,0],f_internal_e[1,0],f_external_e[0,0],f_external_e[1,0],Kte[0,0],Kte[0,1],Kte[1,0],Kte[1,1],sep=',', file=f)
                print(E,v,element_length,p,f_internal_e[0,0],f_internal_e[1,0],f_external_e[0,0],f_external_e[1,0],Kte[0,0],Kte[0,1],Kte[1,0],Kte[1,1],sep=',', file=f)
                #print(E,v,element_length,p,np.format_float_scientific(f_internal_e[0,0]),np.format_float_scientific(f_internal_e[1,0]),sep=',', file=f)
                
                f.close()
                #print('total time =', elaspsed_time)
                #print('count:  ', count, 'element: ', e, 'f_int:  ', f_internal_e.ravel(), 'f_ext:  ', f_external_e.ravel(), 'Kte:   ', Kte.ravel(), '\n')
                count+=1
            delta_uk = -1 *(np.linalg.inv(Kt) @ Rk)
            um = uk + delta_uk  
            nr = nr +1
            
            
            if not Infinity_norm(delta_uk) < 0.005*Infinity_norm(um) or Infinity_norm(F_int-F_ext) < 0.005 * Infinity_norm(F_int): ##Exit statement as given in question
                break

        uvalues[:,m] = np.reshape(um, num_of_elements+1)  ##array of displacement values in each time step
        m = m +1
    
    #f = open('results.txt', 'w')  ##writing to output file

    #print("At final loading(tf):\n\n", "Stresses\n\n\nRadial Stress(Sigma_rr):\n\n" ,stress_values[1:,0], "\n\nCircumfrential Stress(Sigma_phi):\n\n", stress_values[1:,1],  file=f)
    #print("\n\nDisplacements\n\nU(r):\n ", um.T, file=f)
    if Q > 0:  ##for plotting purposes
        type_ = 'Viscoelastic'
    else:
        type_ = 'Elastic'

    #print(nr)
    u_analytical = analytical_elastic(E,v,p,a,b,mesh(a,b,num_of_elements, 2))  ##analytical solution
    #print(u_analytical)
    #print((u_analytical-um.T))
    if (abs(u_analytical-um.T)).max() > 1e-10:
        print('inaccurate solution', abs(u_analytical-um.T).max())
        
    # fig , ax = plt.subplots()
    # ax.scatter(mesh(a,b,num_of_elements, mesh_refinement_factor), u_analytical,marker='*', c='green', label="analytical")
    # ax.plot(rnodes, um, marker='.', color='red',label="fea")
    # ax.set_xlabel('Radius(r)')
    # ax.set_ylabel('Displacement(u)')
    # plt.text(60, 0.047, f'Time Interval(s)={delta_t}')
    # plt.text(60, 0.049, f'Number of elements={num_of_elements}')
    # plt.title(f'Convergence - {type_}')
    # plt.legend()
    # plt.savefig('convergence.png')
 

    return um, rnodes, delta_t , num_of_elements, u_analytical, nr, uvalues



num_of_elements=3
delta_t = 3
#convergence_elastic_study()
#convergence_viscoelastic_study()
#pipe_widening(delta_t, num_of_elements)

#main(E, v, Q*0, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)  ##elastic
#main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)  ##viscoelastic
#f.write('E,v,element_length,Q,T,p,f_int[0],f_int[1],f_ext[0],f_ext[1],kte[00],kte[01],kte[10],kte[11]\n')

f= open('data_linear_elastic.txt', 'w')
f.truncate(0)
f.write('E,v,element_length,p,f_int[0],f_int[1],f_ext[0],f_ext[1],kte[00],kte[01],kte[10],kte[11]\n')
#f.write('E,v,element_length,p,f_int[0],f_int[1]\n')
#f.write('E,v,element_length,Q,T,p,f_int[0],f_int[1],f_ext[0],f_ext[1],kte[00],kte[01],kte[10],kte[11]\n')

f.close()
i = 0
from feamal.data_prep import *
dimensionSpans = np.asarray([[45000,450000], [0.2,0.5]])
Samples = LHCSampling(2,2,500, dimensionSpans,plot=False)
for E,v in zip(Samples[0,:],Samples[1,:]):

    main(E, v, Q*0, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)