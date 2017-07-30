import numpy as np
import psi4


def scf(geom, basis, nel):
    mol = psi4.geometry(geom)
    mol.update_geometry()
    mol.print_out()

    bas = psi4.core.BasisSet.build(mol, target=basis)
    bas.print_out()

    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()

    if (nbf > 100):
        raise Exception("More than 100 basis functions!")

    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())
    E_nuc = mol.nuclear_repulsion_energy()

    H = V + T

    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)


    def diag(F, A):
        Fp = A.T @ F @ A
        eps, Cp = np.linalg.eigh(Fp)
        C = A @ Cp
        return eps, C

    eps, C = diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

    damp_value = 0.20
    damp_start = 5
    e_conv = 1.e-6
    d_conv = 1.e-6
    E_old = 0.0
    F_old = None

    for iteration in range(25):
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)

        F_new = H + 2.0 * J - K
        # conditional iteration > start_damp
        if iteration >= damp_start:
            F = damp_value * F_old + (1.0 - damp_value) * F_new
        else:
            F = F_new

        F_old = F_new
        # F = (damp_value) Fold + (??) Fnew

        # Build the AO gradient
        grad = F @ D @ S - S @ D @ F

        grad_rms = np.mean(grad**2)**0.5

        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + E_nuc

        E_diff = E_total - E_old
        E_old = E_total
        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
              (iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (grad_rms < d_conv):
            break

        eps, C = diag(F, A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T

    return E_total
