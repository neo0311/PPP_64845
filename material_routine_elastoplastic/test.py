import numpy as np
def full_3x3x3x3_to_Voigt_6x6(C):
    """
    Convert from the full 3x3x3x3 representation of the stiffness matrix
    to the representation in Voigt notation. Checks symmetry in that process.
    """

    tol = 1e-3
    Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)] 
    C = np.asarray(C)
    Voigt = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i,j] = C[k,l,m,n]

            #print '---'
            #print k,l,m,n, C[k,l,m,n]
            #print m,n,k,l, C[m,n,k,l]
            #print l,k,m,n, C[l,k,m,n]
            #print k,l,n,m, C[k,l,n,m]
            #print m,n,l,k, C[m,n,l,k]
            #print n,m,k,l, C[n,m,k,l]
            #print l,k,n,m, C[l,k,n,m]
            #print n,m,l,k, C[n,m,l,k]
            #print '---'
    return Voigt



def voigt(X):
    """
    Returns voigt notation of the given tensor
    """
    if np.ndim(X)==4:
        C = X
        C_voigt = np.zeros((6,6))
        m = (np.asarray([1,2,3,2,3,1]) - np.ones(6)).astype(int)
        n = (np.asarray([1,2,3,3,1,2]) - np.ones(6)).astype(int)
        print(m)
        print(n)
        for i in range(6):
            for j in range(6):
                if i == j:
                    C_voigt[i,j] = C[m[i],n[i],m[j],n[j]]
                else:
                    C_voigt[i,j] = C[m[i],n[i],m[j],n[j]]

                    C_voigt[i,j] = C[n[i],m[i],n[j],m[j]]
                    C_voigt[i,j] = C[m[i],n[i],n[j],m[j]]
                    C_voigt[i,j] = C[n[i],m[i],m[j],n[j]]
                #C_voigt[j,i] =C_voigt[i,j]
        print(C_voigt)



def voigt_transform(X):
    #if np.shape(X).all() == np.asarray((3,3,3,3)).all():
    if np.array_equal(np.shape(X), np.asarray((3,3,3,3)))==True:

        tol = 1e-3
        Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)] 
        C = np.asarray(X)
        Voigt = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                k, l = Voigt_notation[i]
                m, n = Voigt_notation[j]
                Voigt[i,j] = C[k,l,m,n]

    elif np.array_equal(np.shape(X), np.asarray([3,3]))==True:
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


C =np.random.randint(0,100,size=(3))
#print(C[0,1,1,2])
#print(C[1,2,1,2])
#print(C[2,0,1,1])

a = np.ones((3,3))
print(np.size(a))