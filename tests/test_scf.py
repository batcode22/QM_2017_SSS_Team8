"""
test for scf
"""

import projects
import numpy as np
import psi4
import pytest


def test_scf():
    psi4.set_options({"scf_type": "pk"})
    psi4_basis_scf = "SCF/" + projects.scf.basis
    psi4_energy_scf = psi4.energy(psi4_basis_scf, molecule=projects.scf.mol)
    E_total_scf = projects.scf.E_total
    assert np.allclose(psi4_energy_scf, E_total_scf)


#def test_diis():
#    psi4.set_options({"scf_type": "pk"})
#    psi4_basis_diis = "SCF/" + projects.diis.basis
#    psi4_energy_diis = psi4.energy(psi4_basis_diis, molecule=projects.diis.mol)
#    E_total = projects.diis.E_total
	#assert np.allclose(psi4_energy_diis, E_total_diis)
	#assert # Something about number of iterations
