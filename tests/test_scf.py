"""
test for math
"""

import projects
import numpy as np
import psi4
import pytest


def test_scf():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    basis = projects.scf.basis
    psi4.set_options({"scf_type": "pk"})
    psi4_basis = "SCF/" + basis
    psi4_energy = psi4.energy(psi4_basis, molecule=mol)
    E_total = projects.scf.E_total
    assert np.allclose(psi4_energy, E_total)
	# assert True
    # assert projects.scf.diag(2, 5) == 7
