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

    ue = np.zeros((2,1))
    um = np.zeros((num_of_elements+1,1))
    uk = np.zeros((num_of_elements+1, 1))
    uvalues = np.zeros((num_of_elements+1, num_of_iterations))
    strain_values = np.zeros((num_of_elements+1, num_of_iterations), dtype=object)
    over_stress_values = np.zeros((num_of_elements+1, num_of_iterations), dtype=object)
    stress_values = np.zeros((num_of_elements+1, 2))
    m = 1
    delta_strain = np.zeros((2,1))

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
                start_time = ti.time()

                ##elemental structural data calculation
                element_length = rnodes[e] - rnodes[e-1]
                element_nodes = np.reshape(np.asanyarray((rnodes[e-1], rnodes[e])),(2,1))
                A = assignment_matrix(e, num_of_elements)
                ue = A@uk

                ##elemental material data calculation
                element_start_time = ti.time()

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
                element_end_time = ti.time()
                print('element_duration', (element_end_time - element_start_time), e)
                Kt = Kt + ((A.T)@(Kte)@(A))
                Rk = Rk + A.T @ (f_internal_e- f_external_e)
                ##Total Forces
                F_ext = F_ext + A.T@(f_external_e)
                F_int = F_int + A.T.dot(f_internal_e)
                elaspsed_time= start_time - ti.time()
                print('total time =', elaspsed_time)
            delta_uk = -1 *(np.linalg.inv(Kt) @ Rk)
            um = uk + delta_uk  
            nr = nr +1

            if not Infinity_norm(delta_uk) < 0.005*Infinity_norm(um) or Infinity_norm(Rk) < 0.005 * Infinity_norm(F_int): ##Exit statement as given in question
                break

        uvalues[:,m] = np.reshape(um, num_of_elements+1)  ##array of displacement values in each time step
        m = m +1
    
    f = open('results.txt', 'w')  ##writing to output file

    print("At final loading(tf):\n\n", "Stresses\n\n\nRadial Stress(Sigma_rr):\n\n" ,stress_values[1:,0], "\n\nCircumfrential Stress(Sigma_phi):\n\n", stress_values[1:,1],  file=f)
    print("\n\nDisplacements\n\nU(r):\n ", um.T, file=f)
    
    if Q > 0:  ##for plotting purposes
        type_ = 'Viscoelastic'
    else:
        type_ = 'Elastic'


    u_analytical = analytical_elastic(E,v,p,a,b,mesh(a,b,20, 2))  ##analytical solution
    
    fig , ax = plt.subplots()
    ax.scatter(mesh(a,b,20, mesh_refinement_factor), u_analytical,marker='*', c='green', label="analytical")
    ax.plot(rnodes, um, marker='+', color='orange',label="fea")
    ax.set_xlabel('Radius(r)')
    ax.set_ylabel('Displacement(u)')
    plt.text(60, 0.047, f'Time Interval(s)={delta_t}')
    plt.text(60, 0.049, f'Number of elements={num_of_elements}')
    plt.title(f'Convergence - {type_}')
    plt.legend()
    plt.savefig('convergence.png')
 

    return um, rnodes, delta_t , num_of_elements, u_analytical, nr, uvalues

def convergence_elastic_study():
    um1, nodes1, delta_t1, num_of_elements1, u_analytical, nr, uvalues=main(E, v, 0, T, a,b,p_max,tL,tF, mesh_refinement_factor, 0.5, 1)
    um2, nodes2, delta_t2, num_of_elements2, u_analytical, nr, uvalues=main(E, v, 0, T, a,b,p_max,tL,tF, mesh_refinement_factor, 0.5, 2)
    um3, nodes3, delta_t3, num_of_elements3, u_analytical, nr, uvalues=main(E, v, 0, T, a,b,p_max,tL,tF, mesh_refinement_factor, 0.5, 5)
    um4, nodes4, delta_t4, num_of_elements4, u_analytical, nr, uvalues=main(E, v, 0, T, a,b,p_max,tL,tF, mesh_refinement_factor, 0.5, 10)

    fig,ax = plt.subplots()
    ax.scatter(mesh(a,b,20, 2), u_analytical,marker='*', c='black', label="analytical")
    ax.plot(nodes1, um1, '--' ,label=f'time interval={delta_t1}, #elements={num_of_elements1}')
    ax.plot(nodes2, um2,'--', label=f'time interval={delta_t2}, #elements={num_of_elements2}')
    ax.plot(nodes3, um3, '--', label=f'time interval={delta_t3}, #elements={num_of_elements3}')
    ax.plot(nodes4, um4, label=f'time interval={delta_t4}, #elements={num_of_elements4}')
    plt.text(56, 0.0465, f'#NR runs for each time step={nr}')
    ax.set_xlabel('Radius(r)')
    ax.set_ylabel('Displacement(u)')
    plt.legend()
    plt.title('Elastic')
    plt.savefig('convergence_study_elastic.png')

def convergence_viscoelastic_study():
    um1, nodes1, delta_t1, num_of_elements1, u_analytical, nr, uvalues=main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, 10, 2)
    um2, nodes2, delta_t2, num_of_elements2, u_analytical, nr, uvalues=main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, 8, 4)
    um3, nodes3, delta_t3, num_of_elements3, u_analytical, nr, uvalues=main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, 5, 5)
    um4, nodes4, delta_t4, num_of_elements4, u_analytical, nr, uvalues=main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, 2, 15)

    fig,ax = plt.subplots()
    ax.scatter(mesh(a,b,20, 2), u_analytical, c='black',marker='*', label="analytical")
    ax.plot(nodes1, um1, '--' ,label=f'time interval={delta_t1}, #elements={num_of_elements1}')
    ax.plot(nodes2, um2, '--', label=f'time interval={delta_t2}, #elements={num_of_elements2}')
    ax.plot(nodes3, um3, '--', label=f'time interval={delta_t3}, #elements={num_of_elements3}')
    ax.plot(nodes4, um4, marker='+',label=f'time interval={delta_t4}, #elements={num_of_elements4}')
    ax.set_xlabel('Radius(r)')
    ax.set_ylabel('Displacement(u)')
    plt.legend()
    plt.title('Viscoelastic')
    plt.savefig('convergence_study_viscoelastic.png')

def pipe_widening(delta_t, num_of_elements): ##plo time history of pipe widening
    um_elastic, nodes, delta_t_elastic, num_of_elements_elastic, u_analytical, nr, uvalues_elastic=main(E, v, 0, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)
    um, nodes, delta_t, num_of_elements, u_analytical, nr, uvalues=main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)
    fig, ax2 = plt.subplots()
    ax2.plot(np.arange(0,tF+delta_t,delta_t), uvalues[-1,:], label="Viscoelastic", color='orange')
    ax2.plot(np.arange(0,tF+delta_t,delta_t), uvalues_elastic[-1,:],'--', label="Elastic", color='black')
    ax2.set_xlabel('Time(t)')
    ax2.set_ylabel('Displacement(u)')
    plt.text(10, 0.007, f'Time Interval(s)={delta_t}')
    plt.text(10, 0.009, f'Number of elements={num_of_elements}')
    plt.title('Time history of pipe widening')
    plt.legend()
    plt.savefig('widening.png')


num_of_elements=50
delta_t = 2
#convergence_elastic_study()
#convergence_viscoelastic_study()
#pipe_widening(delta_t, num_of_elements)

main(E, v, Q*0, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)  ##elastic
#main(E, v, Q, T, a,b,p_max,tL,tF, mesh_refinement_factor, delta_t, num_of_elements)  ##viscoelastic

