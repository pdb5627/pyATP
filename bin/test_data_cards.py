#! /usr/bin/env python

from __future__ import print_function, unicode_literals

import sys
import pyATP

dat_lines = '''BEGIN NEW DATA CASE
LINE CONSTANTS
$ERASE
BRANCH  IN___AOUT__AIN___BOUT__BIN___COUT__C
ENGLISH
  3  0.0   .1357 0   .3959    1.18      5.     42.     30.
  1  0.0   .1357 0   .3959    1.18     -5.     49.     37.
  2  0.0   .1357 0   .3959    1.18      5.     56.     44.
  0  0.0   .6609 0   .4883    .551     -.5     65.     58.
BLANK CARD ENDING CONDUCTOR CARDS
     50.       60.           000001 001000 0    5.98     0        44
$PUNCH
BLANK CARD ENDING FREQUENCY CARDS
BLANK CARD ENDING LINE CONSTANT
BEGIN NEW DATA CASE
BLANK CARD
'''.split('\n')

'''
conductor_card = pyATP.DataCard('(I3, F5.4, F8.5, I2, F8.5, F8.5, F8.3, F8.3, F8.3)',
                                 ['IP', 'SKIN', 'RESIS', 'IX', 'REACT', 'DIAM', 'HORIZ', 'VTOWER', 'VMID'])
end_conductors = pyATP.DataCardFixedText('BLANK CARD ENDING CONDUCTOR CARDS')
conductor_cards = pyATP.DataCardRepeat(conductor_card, end_record = end_conductors)

line_cards = pyATP.DataCardStack([
                        pyATP.DataCardFixedText('BEGIN NEW DATA CASE'),
                        pyATP.DataCardFixedText('LINE CONSTANTS'),
                        pyATP.DataCardFixedText('$ERASE'),
                        pyATP.DataCard('(A8,6A6)', ['branch_card', 'in1', 'out1', 'in2', 'out2', 'in3', 'out3']),
                        pyATP.DataCard('(A80)', ['units']),
                        conductor_cards,
                        pyATP.DataCard('(F8.2, F10.2, A10, A1, 6I1, A1, 6I1, A1, I1, F8.3, A1, 4I1, I1, A7, I3)',
                                       ['RHO', 'FREQ', 'FCAR', '_1',
                                        'inv_C', 'inv_Ce', 'inv_Cs',
                                        'C', 'Ce', 'Cs', '_2',
                                        'Z', 'Ze', 'Zs',
                                        'inv_Z', 'inv_Ze', 'inv_Zg', '_3', 
                                        'ICAP',
                                        'DIST', '_4',
                                        'pi_Y', 'pi_Ys', 'pi_Z', 'pi_Zs',
                                        'ISEG', '_5',
                                        'PUN']),
                        pyATP.DataCardFixedText('$PUNCH'),
                        pyATP.DataCardFixedText('BLANK CARD ENDING FREQUENCY CARDS'),
                        pyATP.DataCardFixedText('BLANK CARD ENDING LINE CONSTANT'),
                        pyATP.DataCardFixedText('BEGIN NEW DATA CASE'),
                        pyATP.DataCardFixedText('BLANK CARD')
                    ])
'''
line_cards = pyATP.LineConstCards()
print('Match? ', line_cards.match(dat_lines))
line_cards.read(dat_lines)
print(line_cards.data)
print('-'*80)

print(line_cards.write())
