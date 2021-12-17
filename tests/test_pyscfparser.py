#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import pytest
import numpy as np
try:
    import pyscf
    from pyscf import dft as pyscfdft, mp as pyscfmp
except Exception:
    pyscf = None

from nomad.datamodel import EntryArchive
from pyscfparser import PySCFParser, pyscf_to_archive


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return PySCFParser()


def test_hf(parser):
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz',symmetry=True)
    mf = mol.HF()
    mf.kernel()
    archive = pyscf_to_archive([mf], filename='temp.json')

    archive = EntryArchive()
    parser.parse('temp.json', archive, None)


def test_dft(parser):
    mol = pyscf.gto.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz', symmetry=True)
    scf = pyscfdft.RKS(mol)
    scf.xc = 'lda,vwn'
    dft = scf.newton() # second-order algortihm
    dft.kernel()

    archive = pyscf_to_archive([dft], filename='temp.json')
    parser.parse('temp.json', archive, None)

    assert archive.run[-1].method[-1].dft.xc_functional.contributions[0].name == 'SLATER'


def test_mp2(parser):
    mol = pyscf.gto.M(atom='O 0 0 0; O 0 0 1.2', basis='ccpvdz', spin=2)
    hf = pyscf.scf.HF(mol).run()
    mp2 = pyscfmp.MP2(hf).run()

    archive = pyscf_to_archive([hf, mp2], filename='temp.json')
    parser.parse('temp.json', archive, None)