import scipy

def soscf(C, F, J, K, nel):
    """  
    Update the density matrix using quasi second-order optimization.

    C   - Orbital coefficients
    F   - Fock matrix
    J   - Coulomb matrix
    K   - Exchange matrix
    nel - Number of electrons

    Return - Density matrix
    """

    # Occupied and virtal block of C and F
    Cocc = C[:, :nel]
    Cvir = C[:, nel+1:]    
    Focc =  Cocc.T @ F @ Cocc
    Fvir =  Cvir.T @ F @ Cvir

    # Orbital rotation parameters.    
    kappa = np.ones(nel * (nbf - nel))
    kappa.reshape(nel, nbf - nel)
    
    epsilon = 0.001
    maxit = 20
    for microiteration in range(maxit):
        kappa_old = kappa        
        theta = Cocc @ kappa @ Cvir.T
        Jtheta = np.einsum("pqrs,rs->pq", g, theta)
        Ktheta = np.einsum("prqs,rs->pq", g, theta)    
        # Hessian * rot.parameters
        kappa = Focc @ kappa - kappa @ Fvir + Cocc.T @ ( 4.0 * Jkappa - Ktheta - Ktheta.T ) @ Cvir
        if(np.mean((kappa - kappa_old) ** 2) ** 0.5 < epsion):
            break

   # Transformaiton from kappa to C
   newkappa = np.zeros(nbf*nbf).reshape(nbf, nbf)
   for p in range(nbf):
       for q in range(nbf):
           if(p <= nel and q > nel):
               newkappa[p, q] = kappa[p, q - nel]
           if(q <= nel and p > nel):
               newkappa[p, q] = kappa[q, p - nel]

   # Return density matrix
   Cnew = scipy.linalg.expm(newkappa)
   Cocc_new = Cnew[:, :nel]
   D = Cocc_new @ Cocc_new.T

   return D
