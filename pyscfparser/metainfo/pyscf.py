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
import numpy as np            # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, Reference, JSON
)

from nomad.datamodel.metainfo import simulation


m_package = Package()


class Scf(simulation.method.Scf):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_init_guess = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_pyscf_diis = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_diis_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_pyscf_diis_start_cycle = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_diis_space = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_diis_space_rollback = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_direct_scf = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_direct_scf_tol = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class BasisSet(simulation.method.BasisSet):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_bas_angular = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_atom = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_coord = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_pyscf_bas_ctr_coeff = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_bas_nprim', 'x_pyscf_bas_nctr'],
        description='''
        ''')

    x_pyscf_bas_exp = Quantity(
        type=np.dtype(np.float64),
        shape=[1],
        description='''
        ''')

    x_pyscf_bas_kappa = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_len_cart = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_len_spinor = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_nctr = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_nprim = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_bas_rcut = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_pyscf_cart = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_cart_labels = Quantity(
        type=str,
        shape=['x_pyscf_bas_len_cart'],
        description='''
        ''')

    x_pyscf_sph_labels = Quantity(
        type=str,
        shape=['x_pyscf_bas_len_cart'],
        description='''
        ''')


class x_pyscf_symm_orb(MSection):
    m_def = Section(validate=False)

    x_pyscf_norb = Quantity(
        type=np.dtype(np.int32),
        shape=[1],
        description='''
        ''')

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_norb'],
        description='''
        ''')


class Method(simulation.method.Method):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_ecp = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_pyscf_has_ecp_soc = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_incore_anyway = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_max_memory = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_pyscf_spin = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_nao = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_nao_2c = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_nao_cart = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_nao_nr = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_nucmod = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_pyscf_nucproc = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_pyscf_omega = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_pyscf_time_reversal_map = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_irrep_nelec = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_pyscf_symm_orb = SubSection(sub_section=x_pyscf_symm_orb.m_def, repeats=True)


class System(simulation.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_inertia_moment = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        description='''
        ''')

    x_pyscf_irrep_id = Quantity(
        type=np.dtype(np.int32),
        shape=[1],
        description='''
        ''')

    x_pyscf_irrep_name = Quantity(
        type=str,
        shape=[1],
        description='''
        ''')

    x_pyscf_groupname = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_pyscf_symmetry = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_pyscf_togroup = Quantity(
        type=str,
        shape=[1],
        description='''
        ''')


class AtomParameters(simulation.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_atom_nshells = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_pyscf_atom_shell_ids = Quantity(
        type=np.dtype(np.int32),
        shape=['x_pyscf_atom_nshells'],
        description='''
        ''')


class BasisSetAtomCentered(simulation.method.BasisSetAtomCentered):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_ao_label = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class BandEnergies(simulation.calculation.BandEnergies):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_mo_coeff = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')


class Calculation(simulation.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_pyscf_hcore = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_fock = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_init_guess = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_j = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_k = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')
    x_pyscf_hcore = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_fock = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_veff = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_stability_internal = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')

    x_pyscf_stability_external = Quantity(
        type=np.dtype(np.float64),
        shape=['x_pyscf_nao', 'x_pyscf_nao'],
        description='''
        ''')


class x_pyscf_io(MSection):
    m_def = Section(validate=False)

    x_yambo_parameters = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')
