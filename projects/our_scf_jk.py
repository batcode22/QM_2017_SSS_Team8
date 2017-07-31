import numpy as np
import psi4

nel = 5


def setup(geom, basis):
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

    return H, A, g, E_nuc, S, mol


def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C

def jk_density_fitting(mol, D, H, basis='aug-cc-pvdz', e_conv = 1e-10, d_conv = 1e-10):

    # Psi4 options
    psi4.set_options({'basis': basis,
                      'scf_type': 'df',
                      'e_convergence': e_conv,
                      'd_convergence': d_conv})

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))

    # Build auxiliary basis set
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", basis)

    # ==> Build Density-Fitted Integrals <==
    # Get orbital basis & build zero basis
    orb = wfn.basisset()
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    # Build instance of MintsHelper
    mints = psi4.core.MintsHelper(orb)

    # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
    Ppq = mints.ao_eri(zero_bas, aux, orb, orb)

    # Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1.e-14)

    # Remove excess dimensions of Ppq, & metric
    Ppq = np.squeeze(Ppq)
    metric = np.squeeze(metric)

    # Build the Qso object
    Qpq = np.einsum('QP,Ppq->Qpq', metric, Ppq)

    # ==> Compute SCF Wavefunction, Density Matrix, & 1-electron H <==
 #   scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

    # Two-step build of J with Qpq and D
    X_Q = np.einsum('Qpq,pq->Q', Qpq, D)
    J = np.einsum('Qpq,Q->pq', Qpq, X_Q)

    # Two-step build of K with Qpq and D
    Z_Qqr = np.einsum('Qrs,sq->Qrq', Qpq, D)
    K = np.einsum('Qpq,Qrq->pr', Qpq, Z_Qqr)

    return J, K

def scf(A,
        H,
        g,
        D,
        E_nuc,
        S,
		mol,
        max_iter=25,
        damp_value=0.20,
        damp_start=5,
        e_conv=1.e-6,
        d_conv=1.e-6, df=False):
    E_old = 0.0

    for iteration in range(max_iter):

        if df==True:
            J, K = jk_density_fitting(mol, D, H)

        else:
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


'''
geom = """
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
"""
basis = 'aug-cc-pVDZ'

H, A, g, E_nuc, S, mol = setup(geom, basis)
eps, C = diag(H, A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T
E_total = scf(A, H, g, D, E_nuc, S, mol, df=True)
psi4.set_options({"scf_type": "df"})
psi4_energy = psi4.energy("SCF/" + basis, molecule=mol)
if np.allclose(psi4_energy, E_total):
	print("It works!")
'''
