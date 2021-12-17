#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import numpy as np
import typing
import json

import pyscfparser
try:
    import pyscf
except Exception:
    pyscf = None

from nomad.datamodel import EntryArchive
from nomad.parsing import FairdiParser
from nomad.units import ureg
from nomad.datamodel.metainfo.simulation.run import (
    Run, Program)
from nomad.datamodel.metainfo.simulation.method import (
    Method, DFT, XCFunctional, Functional, BasisSet, BasisSetAtomCentered, AtomParameters,
    Scf, Electronic
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry, Multipoles, MultipolesEntry, BandEnergies, Charges,
    ChargesValue
)
from pyscfparser.metainfo.pyscf import x_pyscf_symm_orb


class PySCFParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/pyscf', code_name='PySCF',
            code_homepage='https://pyscf.org/',
            mainfile_mime_re=r'(application/json)|(text/.*)',
            mainfile_name_re=r'.*archive_.*\.json',
            mainfile_contents_re=(r'"name": "PySCF"'))

    def parse(self, mainfile: str, archive: EntryArchive, logger=None):
        with open(mainfile, 'rt') as f:
            archive_data = json.load(f)
            archive.m_update_from_dict(archive_data)


class Converter:
    def __init__(self):
        self._archive = None
        self.n_spin = 1
        self.method = None
        self._xc_map = {val: key for key, val in pyscf.dft.libxc.XC_CODES.items()}

    @property
    def archive(self):
        return self._archive

    @archive.setter
    def archive(self, value):
        self.n_spin = 1
        self.method = None
        self._archive = value

    def parse_mol(self, module):
        method = self.archive.run[-1].m_create(Method)
        system = self.archive.run[-1].m_create(System)
        mol = module.mol

        n_spin = 1 if mol.spin == 0 else 2
        self.n_spin = n_spin
        n_elec = [mol.nelectron // n_spin for _ in range(n_spin)]
        method.electronic = Electronic(
            n_spin_channels=n_spin, n_electrons=n_elec, spin_target=mol.multiplicity,
            method=self.method)

        method.x_pyscf_ecp = mol.ecp
        method.x_pyscf_has_ecp_soc = mol.has_ecp_soc()
        method.x_pyscf_incore_anyway = mol.incore_anyway
        method.x_pyscf_max_memory = mol.max_memory
        method.x_pyscf_nao = mol.nao
        method.x_pyscf_nao_sc = mol.nao_2c()
        method.x_pyscf_nao_cart = mol.nao_cart()
        method.x_pyscf_nao_nr = mol.nao_nr()
        method.x_pyscf_nucmod = mol.nucmod
        method.x_pyscf_nucprop = mol.nucprop
        method.x_pyscf_omega = mol.omega
        method.x_pyscf_spin = mol.spin
        method.x_pyscf_time_reversal_map = mol.time_reversal_map()

        if mol.symm_orb is not None:
            for val in mol.symm_orb:
                symm_orb = method.m_create(x_pyscf_symm_orb)
                symm_orb.value = val

        for nbas in range(mol.nbas):
            basis_set = method.m_create(BasisSet)
            # TODO are there other types of basis sets?
            basis_set.type = 'Gaussian'
            basis_set.name = mol.basis
            keys = [
                'angular', 'atom', 'coord', 'ctr_coeff', 'nprim', 'exp', 'kappa',
                'len_cart', 'nctr', 'rcut']
            for key in keys:
                key = 'bas_%s' % key
                if hasattr(mol, key):
                    setattr(basis_set, 'x_pyscf_%s' % key, getattr(mol, key)(nbas))
            # basis_set.x_pyscf_bas_angular = mol.bas_angular(nbas)
            # basis_set.x_pyscf_bas_atom = mol.bas_atom(nbas)
            # basis_set.x_pyscf_bas_coord = mol.bas_coord(nbas)
            # basis_set.x_pyscf_bas_ctr_coeff = mol.bas_ctr_coeff(nbas)
            # basis_set.x_pyscf_bas_exp = mol.bas_exp(nbas)
            # basis_set.x_pyscf_bas_kappa = mol.bas_kappa(nbas)
            # basis_set.x_pyscf_bas_len_cart = mol.bas_len_cart(nbas)
            # basis_set.x_pyscf_bas_len_spinor = mol.bas_len_spinor(nbas)
            # basis_set.x_pyscf_bas_nctr = mol.bas_nctr(nbas)
            # basis_set.x_pyscf_bas_nprim = mol.bas_nprim(nbas)
            # basis_set.x_pyscf_bas_rcut = mol.bas_rcut(nbas)
            basis_set.x_pyscf_cart = mol.cart
            basis_set.x_pyscf_cart_labels = mol.cart_labels()[mol.ao_loc[nbas]: mol.ao_loc[nbas + 1]]
            basis_set.x_pyscf_sph_labels = mol.sph_labels()[mol.ao_loc[nbas]: mol.ao_loc[nbas + 1]]

            for nao in range(mol.ao_loc[nbas], mol.ao_loc[nbas + 1]):
                atom_centered = basis_set.m_create(BasisSetAtomCentered)
                atom_centered.x_pyscf_ao_label = mol.ao_labels()[nao]

        for n in range(mol.natm):
            atom_parameters = method.m_create(AtomParameters)
            atom_parameters.charge = mol.atom_charges()[n] * ureg.elementary_charge
            atom_parameters.mass = mol.atom_mass_list()[n] * ureg.amu
            atom_parameters.n_electrons = mol.nelectron
            atom_parameters.n_core_electrons = mol.atom_nelec_core(n)
            atom_parameters.label = mol.atom_pure_symbol(n)
            atom_parameters.x_pyscf_atom_nshells = mol.atom_nshells(n)
            atom_parameters.x_pyscf_atom_shell_ids = mol.atom_shell_ids(n)


        # mol.energy_nuc

        system.atoms = Atoms(labels=mol.elements, positions=mol.atom_coords() * ureg.bohr)
        if hasattr(mol, 'lattice_vectors'):
            # apparently the unit is not consistent with mol.unit
            system.atoms.lattice_vectors = mol.lattice_vectors() * ureg.bohr

        system.x_pyscf_inertia_moment = mol.inertia_moment()
        system.x_pyscf_irrep_id = mol.irrep_id
        system.x_pyscf_irrep_name = mol.irrep_name
        system.x_pyscf_groupname = mol.groupname
        system.x_pyscf_symmetry = mol.symmetry
        system.x_pyscf_topgroup = mol.topgroup


    def parse_calculation(self, module):
        calculation = self.archive.run[-1].m_create(Calculation)

        # energies
        calculation.energy = Energy(
            total=EnergyEntry(value=module.energy_tot()), contributions=[
                EnergyEntry(kind='electronic', value=module.energy_elec()),
                EnergyEntry(kind='nuclear', value=module.energy_nuc())]
        )
        if hasattr(module, 'get_fermi'):
            calculation.energy.fermi = module.get_fermi() * ureg.hartree

        # forces
        grad = module.nuc_grad_method()
        calculation.forces = Forces(
            total=ForcesEntry(value=grad.grad() * ureg.hartree / ureg.bohr),
            contributions=[
                ForcesEntry(kind='electronic', value=grad.grad_elec()),
                ForcesEntry(kind='nuclear', value=grad.grad_nuc())])

        calculation.calculation_converged = module.converged
        calculation.x_pyscf_init_guess = module.get_init_guess()
        calculation.x_pyscf_j = module.get_j()
        calculation.x_pyscf_k = module.get_k()
        calculation.x_pyscf_hcore = module.get_hcore()
        calculation.x_pyscf_fock = module.get_fock()
        calculation.x_pyscf_veff = module.get_veff()

        # dipoles
        multipoles = calculation.m_create(Multipoles)
        multipoles.dipole = MultipolesEntry(total=module.dip_moment())

        # eigenvalues
        if module.mo_energy is not None:
            eigenvalues = calculation.m_create(BandEnergies)
            shape = (self.n_spin, 1, len(module.mo_energy))
            eigenvalues.energies = np.reshape(module.mo_energy, shape) * ureg.hartree
            eigenvalues.occupations = np.reshape(module.mo_occ, shape)
            eigenvalues.x_pyscf_mo_coeff = module.mo_coeff

        def parse_charges(orbital_charges, atomic_charges, method):
            charges = calculation.m_create(Charges)
            charges.analysis_method = method
            charges.value = atomic_charges * ureg.elementary_charge
            for n, value in enumerate(orbital_charges):
                orbital = charges.m_create(ChargesValue, Charges.orbital_projected)
                orbital.value = value * ureg.elementary_charge
                orbital.orbital = module.mol.ao_labels()[n]

        # mulliken
        orbital_charges, atomic_charges = module.mulliken_pop(verbose=False)
        parse_charges(orbital_charges, atomic_charges, 'Mulliken')
        # loewdin
        orbital_charges, atomic_charges = module.mulliken_pop_meta_lowdin_ao(verbose=False)
        parse_charges(orbital_charges, atomic_charges, 'Loewdin')
        # loewdin-meta
        orbital_charges, atomic_charges = module.mulliken_meta(verbose=False)
        parse_charges(orbital_charges, atomic_charges, 'Loewdin meta')

        stability = module.stability()
        calculation.x_pyscf_stability_internal = stability[0]
        calculation.x_pyscf_stability_external = stability[1]

    def parse_scf(self, module):
        run = self.archive.run[-1]

        if hasattr(module, 'xc'):
            self.method = 'DFT'
        elif hasattr(module, 'emp2'):
            self.method = 'MP2'

        self.parse_mol(module)
        # TODO put this somewhere
        run.method[-1].x_pyscf_irrep_nelec = module.get_irrep_nelec()

        run.method[-1].scf = Scf(
            n_max_iteration=module.max_cycle,
            threshold_energy_change=module.conv_tol, threshold_gradient=module.conv_tol_grad,
            x_pyscf_init_guess=module.init_guess, x_pyscf_diis=module.diis,
            x_pyscf_diis_file=module.diis_file, x_pyscf_diis_space=module.diis_space,
            x_pyscf_diis_space_rollback=module.diis_space_rollback,
            x_pyscf_diis_start_cycle=module.diis_start_cycle, x_pyscf_direct_scf=module.direct_scf,
            x_pyscf_direct_scf_tol=module.direct_scf_tol)

        if hasattr(module, 'xc'):
            run.method[-1].dft = DFT()
            xc_functional = run.method[-1].dft.m_create(XCFunctional)
            for index in pyscf.dft.libxc.parse_xc_name(module.xc):
                name = self._xc_map.get(index)
                if '_X_' in name or name.endswith('_X'):
                    xc_functional.exchange.append(Functional(name=name))
                elif '_C_' in name or name.endswith('_C'):
                    xc_functional.correlation.append(Functional(name=name))
                elif 'HYB' in name:
                    xc_functional.hybrid.append(Functional(name=name))
                else:
                    xc_functional.contributions.append(Functional(name=name))

        self.parse_calculation(module)

    def parse_module(self, module):
        run = self.archive.m_create(Run)
        run.program = Program(name='PySCF', version=pyscf.__version__)
        if hasattr(module, 'scf'):
            self.parse_scf(module)


def pyscf_to_archive(modules: typing.List[typing.Any] = None, filename=None):
    '''
    Converts pyscf modules to the nomad archive format.
    '''
    archive = EntryArchive()
    converter = Converter()
    converter.archive = archive
    for module in modules:
        converter.parse_module(module)

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(archive.m_to_dict(), f, indent=4)

    return archive