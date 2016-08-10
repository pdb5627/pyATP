#! /usr/bin/env python

from __future__ import print_function, unicode_literals

import pyATP
import lineZ
import numpy as np
from numpy.linalg import inv
np.set_printoptions(linewidth=120)

import shutil
import os

ATP_TEST_DIR = r'C:\Users\pdbrown\Documents\ATPdata\work\L1241_Brad'
ATP_TMP_FILENAME = 'test_tmp.atp'

ATP_template = '1240_1241_Phase_Revise2.atp'

current_key = '5432.1'
switch_key = '-7.654'

in_port = ('SRC', 'S0') # Nodes of measuring switch for current
out_port = ('S1', 'TERRA') # Nodes of measuring switch for current

ABCD = pyATP.extract_ABCD(os.path.join(ATP_TEST_DIR, ATP_template),
                          ATP_TMP_FILENAME, current_key, switch_key,
                          in_port, out_port)
A, B, C, D = lineZ.ABCD_breakout(ABCD)

Z_ATP = B # .dot(inv(D))

print('ATP Line Impedance:\n', Z_ATP)
print('Phase impedance imbalance: %.4f %%' % lineZ.impedance_imbalance(Z_ATP))
print('Negative-sequence unbalance factor: %.4f %%' % lineZ.neg_seq_unbalance_factor(Z_ATP))
print('Zero-sequence unbalance factor: %.4f %%' % lineZ.zero_seq_unbalance_factor(Z_ATP))
print('Phase impedances:', np.absolute(Z_ATP.dot(lineZ.Apos)).T)
