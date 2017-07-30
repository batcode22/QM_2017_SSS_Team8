import numpy as np
import psi4

def diis(geom, basis, nel):
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

    max_iter = 25
    damp_value = 0.20
    damp_start = 5
    e_conv = 1.e-9
    d_conv = 1.e-8
    E_old = 0.0
    E_total_diis = 0.0
    F_list = []
    DIIS_grad = []

    for iteration in range(max_iter):
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)

        F = H + 2.0 * J - K

        # Build the DIIS AO gradient
        grad = F @ D @ S - S @ D @ F
        diis_grad = A.T @ grad @ A

        F_list.append(F)
        DIIS_grad.append(diis_grad)

        diis_rms = np.mean(diis_grad**2)**0.5

        # Build the energy
        E_diis = np.sum((F + H) * D)
        E_total_diis = E_diis + E_nuc

        E_diff = E_total_diis - E_old
        E_old = E_total_diis
        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
              (iteration, E_total_diis, E_diff, diis_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (diis_rms < d_conv):
            break

        if iteration >=2:
            dim_B = len(F_list) + 1
            B = np.empty((dim_B, dim_B))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for i in range(len(F_list)):
                for j in range(len(F_list)):
                    B[i, j] = np.einsum('ij,ij->', DIIS_grad[i], DIIS_grad[j])

            right = np.zeros(dim_B)
            right[-1] = -1

            cn = np.linalg.solve(B, right)

            F = np.zeros_like(F)
            for x in range(cn.shape[0] - 1):
                F += cn[x] * F_list[x]


        F_new = A @ F @ A
        eps, Cp = np.linalg.eigh(F_new)
        C = A @ Cp
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T

    return E_total_diis
