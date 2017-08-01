import numpy as np
import psi4


def jk(geom, basis, nel):
    mol = psi4.geometry(geom)
    mol.update_geometry()
    mol.print_out()

    bas = psi4.core.BasisSet.build(mol, target=basis)

    # Build JKFIT basis
    aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=basis)

    # Zero basis set
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

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

    # Build (P|pq) raw ERIs (1, Naux, nbf, nbf)
    Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
    Qls_tilde = np.squeeze(Qls_tilde)

    # Build and invert Coulomb metric (1, Naux, 1, Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1e-14)
    metric = np.squeeze(metric)

    # Get (P|ls)
    pls = np.einsum("qls,pq->pls", Qls_tilde, metric)

    max_iter = 25
    damp_value = 0.20
    damp_start = 5
    e_conv = 1.e-6
    d_conv = 1.e-6
    E_old = 0.0
    F_new = None

    for iteration in range(max_iter):
        # Coulomb matrix
        chi = np.einsum("pls,ls->p", pls, D)
        J = np.einsum("mnp,p->mn", pls.transpose(1,2,0), chi)

        # Exchange matrix
        ex1 = np.einsum("qms,sp->qmp", pls, Cocc)
        ex2 = np.einsum("qnl,lp->qnp", pls, Cocc)
        K = np.einsum("qmp,qnp->mn", ex1, ex2)

        F_new = H + 2.0 * J - K

        # conditional iteration > start_damp
        if iteration >= damp_start:
            F = damp_value * F_old + (1.0 - damp_value) * F_new
        else:
            F = F_new

        F_old = F_new

        # Build the AO gradient
        grad = A.T @ (F @ D @ S - S @ D @ F) @ A

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
