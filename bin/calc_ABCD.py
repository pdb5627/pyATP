#! /usr/bin/env python

from __future__ import print_function, unicode_literals

import pyATP
import lineZ
from lineZ import Polar
import numpy as np
from numpy.linalg import inv
np.set_printoptions(linewidth=120, precision=4)

import os

ATP_TEST_DIR = r'C:\Users\pdbrown\Documents\ATPdata\work\L1241_Brad'
ATP_TMP_FILENAME = 'test_tmp.atp'

ATP_template = '1241_Phase_pdb_Rev1.atp'
#ATP_template = '1241_Phase_Revise2.atp'

current_key = '800.'
switch_key = '-7.654'

in_port = ('SRC', 'S0') # Nodes of measuring switch for current
out_port = ('S1', 'TERRA') # Nodes of measuring switch for current

ABCD = pyATP.extract_ABCD(os.path.join(ATP_TEST_DIR, ATP_template),
                          ATP_TMP_FILENAME, current_key, switch_key,
                          in_port, out_port)
A, B, C, D = lineZ.ABCD_breakout(ABCD)

print('ATP Line Impedance:')
print('ATP File: {}'.format(os.path.join(ATP_TEST_DIR, ATP_template)))
print('')
for include_Y in (True, False):
    if include_Y:
        Z_ATP = B.dot(inv(D))
        print('Including current to shunt capacitance on source end:')
    else:
        Z_ATP = B
        print('Including series impedance only:')
    Z_s = lineZ.ph_to_seq_m(Z_ATP)
    print('Phase impedance unbalance factor: %.4f %%' %
          lineZ.impedance_imbalance(Z_ATP))
    print('Negative-sequence unbalance factor: %.4f %%' %
          lineZ.neg_seq_unbalance_factor(Z_ATP))
    print('Zero-sequence unbalance factor: %.4f %%' %
          lineZ.zero_seq_unbalance_factor(Z_ATP))
    print('Phase impedances:', np.absolute(lineZ.phase_impedances(Z_ATP)))
    print('Z21: {:.4f} Ohms'.format(Polar(Z_s[2, 1])))
    print('Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}' \
          .format(Z_s[1, 1],
                  Z_s[0, 0],
                  Polar(Z_s[2, 1])))
    print('')

print('Full ABCD Transmission Matrix [T]:')
print(ABCD)
print('[V1; I1] = [T]*[V2; I2] where I1 is into the line and I2 is out of the '
      'line.')
