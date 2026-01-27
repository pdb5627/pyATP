from math import sqrt
import numpy as np
import text_data_cards as tdc
from .dtypes import fdtype, cdtype

# Parsing function for reading / modifying line constant cards. See Rule Book
# Chapter 21. Below is an example of such a card for a three-phase line:
'''
BEGIN NEW DATA CASE
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
'''

class LineConstCards(tdc.DataCardStack):
    ''' Stack of cards for a line constants case.
        This is Based on what ATPDraw creates.'''
    def __init__(self):
        conductor_card = tdc.DataCard('(I3, F5.4, F8.5, I2, F8.5, F8.5, F8.3, F8.3, F8.3)',
                                 ['IP', 'SKIN', 'RESIS', 'IX', 'REACT', 'DIAM', 'HORIZ', 'VTOWER', 'VMID'])
        end_conductors = tdc.DataCardFixedText('BLANK CARD ENDING CONDUCTOR CARDS')
        conductor_cards = tdc.DataCardRepeat(conductor_card, end_record = end_conductors,
                                         name = 'conductors')

        tdc.DataCardStack.__init__(self,
            [tdc.DataCardFixedText('BEGIN NEW DATA CASE'),
             tdc.DataCardFixedText('LINE CONSTANTS'),
             tdc.DataCardFixedText('$ERASE'),
             tdc.DataCard('(A8,6A6)', ['branch_card', 'in1', 'out1', 'in2',
                                       'out2', 'in3', 'out3']),
             tdc.DataCard('(A80)', ['units']),
                        conductor_cards,
             tdc.DataCard('(F8.2, F10.2, A10, A1, 6I1, A1, 6I1, A1, I1, F8.3, A1, 4I1, I1, A7, I3)',
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
             tdc.DataCardFixedText('$PUNCH'),
             tdc.DataCardFixedText('BLANK CARD ENDING FREQUENCY CARDS'),
             tdc.DataCardFixedText('BLANK CARD ENDING LINE CONSTANT'),
             tdc.DataCardFixedText('BEGIN NEW DATA CASE'),
             tdc.DataCardFixedText('BLANK CARD')
            ])

comment_card = tdc.DataCard('A2, A78', ['C ', 'Comment'], fixed_fields=(0,))
vintage_card = tdc.DataCard('A9, A71', ['$VINTAGE,', 'Flag'], fixed_fields=(0,))

class UnitsCard(tdc.DataCard):
    def __init__(self, name=None):
        # The Flag section is not fixed width. Instead, it is comma-separated
        super().__init__(
            'A7, A73',
            ['$UNITS,', 'UnitsStr'],
            fixed_fields=(0,),
            name=name,
            post_read_hook=self.get_XOPT_COPT
        )
        self.XOPT, self.COPT = None, None

    def get_XOPT_COPT(self):
        flags = self.data["UnitsStr"].split(",")
        self.data["XOPT"], self.data["COPT"] = float(flags[0]), float(flags[1])




class LineConstPCHCards(tdc.DataCardStack):
    """ Stack of cards output by a line constants case, based on three-phase
        line and the way ATPDraw runs the case. NOTE: L and C parameters will
        be returned in units of Henries and Siemens. Conversion will be based
        the $UNITS card values.
        assumed to be in the file at nominal frequency. For 60 Hz,
        there should be a $UNITS card with values 60., 60.. This object will
        still read the data otherwise, but matrices may have mH or uF instead of
        Ohms and microSiemens.

    """
    def __init__(self):
        end_record = tdc.DataCardAlternates([
            vintage_card,
            UnitsCard(),
        ])

        tdc.DataCardStack.__init__(self,
            [tdc.DataCardRepeat(comment_card),
             tdc.DataCardOptional(vintage_card),
             UnitsCard(name="units"),
             tdc.DataCardOptional(vintage_card),
             tdc.DataCardRepeat(tdc.DataCard('I2, 4A6, 3E16.0',
                     ['PH', 'BUS1', 'BUS2', 'BUS3', 'BUS4', 'R', 'L', 'C']),
                 end_record, name='RLC_params'),
             end_record],
             post_read_hook=self.get_ZY)
        self.Z = None
        self.Y = None

    def get_ZY(self):
        """ Convert R, L, C parameters to Z and Y matrices"""
        # First compute the number of phases
        n = len(self.data['RLC_params'])
        # n = (n_ph+1)*n_ph/2
        # Solve for n_ph using quadratic formula
        n_ph = int((sqrt(1+8*n) - 1)/2)

        # Conversion factors
        if self.data["units"]["XOPT"] > 0:
            # Xl is given in Ohms: Xl = ω*L => L = Xl / ω
            L_conversion = 1 / (2 * np.pi * self.data["units"]["XOPT"])
        else:
            # L is given in mH
            L_conversion = 1e-3
        if self.data["units"]["COPT"] > 0:
            # Bc is given in μS: Bc = ω*C => C = Bc / ω
            C_conversion = 1e-6 / (2 * np.pi * self.data["units"]["COPT"])
        else:
            # C is given in μF
            C_conversion = 1e-6

        Z = np.zeros((n_ph, n_ph), dtype=cdtype)
        Y = np.zeros((n_ph, n_ph), dtype=cdtype)
        idx = 0
        for r in range(n_ph):
            for c in range(r+1):
                Z[r, c] = Z[c, r] = (self.data['RLC_params'][idx]['R'] +
                                     1j*L_conversion*self.data['RLC_params'][idx]['L'])
                # C assumed to be in microSiemens. (Ref R.B. IV.B.3)
                # C value is total capacitance, which ATP then divides by
                # two for the pi model.
                Y[r, c] = Y[c, r] = 1j*C_conversion*self.data['RLC_params'][idx]['C']
                idx += 1
        self.Z = Z
        self.Y = Y
        return Z, Y
