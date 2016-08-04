#! /usr/bin/env python

from __future__ import print_function, unicode_literals

import pyATP
import lineZ
import numpy as np
from numpy.linalg import inv
np.set_printoptions(linewidth=120)

import re, sys, shutil

ATP_template = 'RPlanPhasingGT_pdb.atp' # Calcs below key on the filename to use right parameters
ATP_tmp = 'test2.atp'
current_key = '866.025'
switch_key = '-7.654'
test_current = 500.
phases =  ('A', 'B', 'C')

in_port = ('SRC', 'S0') # Nodes of measuring switch for current
out_port = ('S1', 'TERRA') # Nodes of measuring switch for current

ABCD = pyATP.extract_ABCD(ATP_template, ATP_tmp, current_key, switch_key,
                          in_port, out_port, test_current = test_current,
                          phases = phases)
A, B, C, D = lineZ.ABCD_breakout(ABCD)

# Validation case w/ positive-sequence balanced current injection
shutil.copyfile(ATP_template, ATP_tmp)
# Find/replace code numbers in template ATP file and copy to new file
pyATP.replace_text(ATP_tmp, current_key, ('%6f.' % test_current))
pyATP.replace_text(ATP_tmp, switch_key, '-1')

# Run ATP on new file
pyATP.run_ATP(ATP_tmp)

# Extract steady-state results
LIS_results = pyATP.lis_filename(ATP_tmp)
node_voltages, branch_currents = pyATP.get_SS_results(LIS_results)

V1_s_test = np.zeros(3, dtype=np.complex128)
I1_s_test = np.zeros(3, dtype=np.complex128)
I2_s_test = np.zeros(3, dtype=np.complex128)

for n2, ph2 in enumerate(phases):
    V1_s_test[n2] = node_voltages[pyATP.node_ph(in_port[0], ph2)]
    I1_s_test[n2] = branch_currents[(pyATP.node_ph(in_port[0], ph2),
                        pyATP.node_ph(in_port[1], ph2))]
    I2_s_test[n2] = branch_currents[(pyATP.node_ph(out_port[0], ph2),
                        pyATP.node_ph(out_port[1], ph2))] # removed +ph2 because 'TERRA'
print('Test Case: Balanced 3-phase injection, output shorted')
print('Injected currents: ', I1_s_test)
print('Source voltages: ', V1_s_test)
I2_s_calc = inv(D).dot(I1_s_test)
V1_s_calc = B.dot(I2_s_calc)
print('Calculated source voltages: ', V1_s_calc)
print('Calculated phase impedance magnitudes:', np.abs(V1_s_test/I1_s_test))

print('A:', A)
print('B:', B)
print('C:', C)
print('D:', D)

Z_ATP = B.dot(inv(D))

# Test data
# Phase impedance matrices of the three construction types, in Ohms/mile
Zstr = {}
Pos = {}
Ystr = {}

# Lattice Structure. Phase order is center, left (under OPGW), right (under EHS).
Zstr['L'] = np.matrix([[0.1640892 + 0.9472117j, 0.1139189 + 0.3632402j, 0.1128743 + 0.3742991j],
                       [0.1139189 + 0.3632402j, 0.1624559 + 0.9440421j, 0.1115062 + 0.2889892j],
                       [0.1128743 + 0.3742991j, 0.1115062 + 0.2889892j, 0.1605425 + 0.9655567j]]) # From ATP

Ystr['L'] = np.matrix([[0 +  6.254276j, 0 + -1.145664j,  0 + -1.147074j],
                       [0 + -1.145664j, 0 +  6.030004j,  0 + -0.3833229j],
                       [0 + -1.147074j, 0 + -0.3833229j, 0 +  6.027235j]])*1.E-6 # From ATP

Pos['L'] = ['Center', 'OPGW side', 'EHS side']

# Single-Pole (delta) Structure. Phase order is bottom, center, top.
# The EHS shield wire is assumed to be on the side with two phases, and the OPGW is on
# the side with only one phase.
Zstr['SP'] = np.matrix([[0.1525965 + 0.9640908j, 0.1067041 + 0.3640585j, 0.1094727 + 0.3759902j],
                        [0.1067041 + 0.3640585j, 0.1580240 + 0.9448685j, 0.1127501 + 0.3564543j],
                        [0.1094727 + 0.3759903j, 0.1127501 + 0.3564543j, 0.1646833 + 0.9292292j]]) # From ATP RP_S1

Ystr['SP'] = np.matrix([[0 +  6.260569j, 0 + -0.9819296j, 0 + -1.215984j],
                        [0 + -0.9819296j, 0 +  6.128009j, 0 + -1.122124j],
                        [0 + -1.215984j, 0 + -1.122124j, 0 +  6.211305j]])*1.E-6 # From ATP RP_S1

Pos['SP'] = ['Bot', 'Mid', 'Top']

if ATP_template == 'RPlanPhasingGT_pdb.atp':
    ######################
    # L3525 Calcs
    # Copied from ipython notebook
    ######################
    # Segment lengths & Configurations
    # PIs is an array of the mileage at the transition points (PIs) measured approximately the route in GIS
    t1 = (41.181 + 20.236)/2
    t2 = (93.8 + 54.561*2)/3
    t3 = (93.8*2 + 54.561)/3

    sections = [(0,      'SP', 'GGS Sub'),
                (11.144, 'SP', 'Heavy Angle Deadend'),
                (12.248, 'SP', 'Heavy Angle Deadend'),
                (14.6,   'L',  'Transition'),
                (t1,     'L',  'Storm Structure'),
                (49.8  , 'SP', 'Transition'),
                (52.097, 'SP', 'Heavy Angle Deadend'),
                (52.558, 'SP', 'Heavy Angle Deadend'),
                (54.561, 'SP', 'Heavy Angle Deadend'),
                (t2,     'SP', 'Storm Structure'),
                (t3,     'SP', 'Storm Structure'),
                (93.8,   'L',  'Transition'),
                (100.9,  '',   'Thedford Sub')]

    PIs = np.array([s[0] for s in sections])
    PI_desc = [s[2] for s in sections]

    # L is an array of the lengths of the segments
    L = PIs[1:] - PIs[:-1]

    # SP  : Single pole
    # L   : Lattice
    # _ after structure type indicates that phase order should be maintained to following section
    str_types = [s[1] for s in sections[:-1]]

    assert(len(L) == len(str_types)) # Just to make sure lengths match, at least

    Pt_sel = ((0, 1, 2), (0, 1, 2), (0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (1, 0, 2), (0, 1, 2), (2, 1, 0))

else:
    ######################
    # L3526 Calcs
    # Copied from ipython notebook
    ######################
    # Segment lengths & Configurations
    # PIs is an array of the mileage at the transition points (PIs) measured approximately the route in GIS
    t1 = (26.271 + 3.773)/2
    t2 = (66.066 + 40.7)/2
    t3 = (93.233 + 66.591)/2


    sections = [(0,      'L',  'Thedford Sub'),
                (t1,     'L',  'Storm Structure'),
                (35.6,   'SP', 'Transition'),
                (40.7,   'L',  'Transition'),
                (t2,     'L',  'Storm Structure'),
                (t3  ,   'L',  'Storm Structure'),
                (94.6,   'SP', 'Transition'),
                (103.933,'SP', 'Heavy Angle Deadend'),
                (104.523,'SP', 'Heavy Angle Deadend'),
                (108.46, 'SP', 'Heavy Angle Deadend'),
                (108.996,'SP', 'Heavy Angle Deadend'),
                (117.759,'SP', 'Heavy Angle Deadend'),
                (118.25, 'SP', 'Heavy Angle Deadend'),
                (121.254,'SP', 'Heavy Angle Deadend'),
                (121.772,'SP', 'Heavy Angle Deadend'),
                (124.1,  '', 'Holt County Sub')]

    PIs = np.array([s[0] for s in sections])
    PI_desc = [s[2] for s in sections]
    
    # L is an array of the lengths of the segments
    L = PIs[1:] - PIs[:-1]

    # SP  : Single pole
    # L   : Lattice
    # _ after structure type indicates that phase order should be maintained to following section
    str_types = [s[1] for s in sections[:-1]]

    assert(len(L) == len(str_types)) # Just to make sure lengths match, at least
    
    Pt_sel = ((0, 1, 2), (1, 0, 2), (0, 1, 2), (0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 1, 2), (0, 2, 1), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2))

    ######################


hyperbolic = True # False for pi model in ATP, True for Bergeron model. Only small difference for short sections
                  
# Verify impedance calcs and results
Z_sel = lineZ.impedance_calcs(Zstr,  Ystr, L, str_types, Pt_sel, print_calcs=False, hyperbolic=hyperbolic, shunt=True, Z_w_shunt=True)

print('ATP Z_sel:\n', Z_ATP)
print('Phase impedance imbalance: %.4f %%' % lineZ.impedance_imbalance(Z_ATP))
print('Negative-sequence unbalance factor: %.4f %%' % lineZ.neg_seq_unbalance_factor(Z_ATP))
print('Zero-sequence unbalance factor: %.4f %%' % lineZ.zero_seq_unbalance_factor(Z_ATP))
print('Phase impedances:', np.absolute(Z_ATP.dot(lineZ.Apos)).T)

print('impedance_calcs Z_sel:\n', Z_sel)
print('Phase impedance imbalance: %.4f %%' % lineZ.impedance_imbalance(Z_sel))
print('Negative-sequence unbalance factor: %.4f %%' % lineZ.neg_seq_unbalance_factor(Z_sel))
print('Zero-sequence unbalance factor: %.4f %%' % lineZ.zero_seq_unbalance_factor(Z_sel))
print('Phase impedances:', np.absolute(Z_sel.dot(lineZ.Apos)).T)

