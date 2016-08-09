#! /usr/bin/env python
# -- coding: utf-8 --

from __future__ import print_function, unicode_literals

import os
import pyATP
import lineZ
from lineZ import cdtype
import cmath
import itertools
import pickle
import argparse

import time

time_fun = time.clock

import sys, shutil, glob, itertools, codecs, os, math
import numpy as np
np.set_printoptions(linewidth=120)
from matplotlib import pyplot as plt

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def polar_formatter(x):
    m = np.absolute(x)
    a = np.angle(x, deg=True)
    return str(m) + ' @ ' + str(a)

class Polar(complex):
    def __format__(self, format_spec):
        m = abs(self)
        a = math.degrees(cmath.phase(self))
        return ('{:' + format_spec + '} @ {: ' + format_spec + '}').format(m, a)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('binary_met_data',
                    type=argparse.FileType('rb'),
                    help='Binary file of extracted meter data in Python '
                         'pickle format. For format, see extract_met_data in '
                         'sel_utilities package.')
parser.add_argument('atp_pch_folder',
                    help='Folder in which ATP pch files of line parameters '
                         'are found.')
parser.add_argument('binary_atp_data',
                    type=argparse.FileType('rb'),
                    help='Name of binary (Python pickle format) '
                         'file to read. See make_ss_csv.py.')


def terminal_state(line_defs, line, terminal=(0, 1)):
    """ Extract phase voltages and currents from meter data in the line_defs
        dict.
    """
    rtn = []
    for n in terminal:
        s = 'term%d_met' % n
        t = line_defs[line][s]
        V_ph = np.array(
            [t['V%s' % ph] for ph in ['A', 'B', 'C']]) * 1e3  # kV to V
        I_ph = np.array(
            [t['I%s' % ph] for ph in ['A', 'B', 'C']])  # already A
        # Reverse polarity of I1
        if n == 0:
            I_ph *= -1
        X_ph = np.concatenate((V_ph, I_ph))
        rtn.append(X_ph)
    return tuple(rtn)


def set_angles(line_defs, bus, buses_set=None):
    # On first call, set the bus to be reference
    if buses_set is None:
        buses_set = {bus: 12.35}

    for line, l in line_defs.items():
        if bus == l['terminals'][0]:
            term = 0
        elif bus == l['terminals'][1]:
            term = 1
        else:
            continue

        rem_bus = l['terminals'][1 - term]
        # Only set each bus angle once
        if rem_bus in buses_set:
            continue

        # Calculate remote bus angle from three-phase power and
        # positive-sequence  line parameters.
        try:
            V1_mag = 1000. * l['term%d_met' % term]['V1_MAG']
        except KeyError:
            try:
                V1_mag = 1000. * l['term%d_met' % term]['VA_MAG']
            except KeyError:
                V1_mag = 115e3 / math.sqrt(3)
        try:
            P = l['term%d_met' % term]['MW_3P']
            Q = l['term%d_met' % term]['MVAR_3P']
        except KeyError:
            P = -1*l['term%d_met' % (1 - term)]['MW_3P']
            Q = -1*l['term%d_met' % (1 - term)]['MVAR_3P']
        ABCD_s = l['ABCD_s']
        A, B, C, D = lineZ.ABCD_breakout(ABCD_s)
        I = (P - 1j*Q)*1e6 / 3. / V1_mag
        Vr = A[1, 1]*V1_mag - B[1, 1]*I
        buses_set[rem_bus] = math.degrees(cmath.phase(Vr)) + buses_set[bus]
        set_angles(line_defs, rem_bus, buses_set)

    return buses_set

def main(argv=None):
    start_time = time_fun()
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    # Read meter data from pickle file.
    with args.binary_met_data as picklefile:
        data_list = pickle.load(picklefile)

    # Read ATP steady-state data from pickle file.
    with args.binary_atp_data as picklefile:
        atp_data_list = pickle.load(picklefile)

    line_defs = {'L1078B': {'terminals': ('Thedford', 'Stapleton'),
                            'atp_segs': ('L78BB', 'L78BA'),
                            'atp_branches': (('THEDF', 'THSTA'),
                                             ('STAPL', 'STATH'))},
                 'L1078A': {'terminals': ('Stapleton', 'Maxwell'),
                            'atp_segs': ('L78AB', 'L78AA'),
                            'atp_branches': (('STAPL', 'STMAX'),
                                             ('MAXWE', 'MAXST'))},
                 'L1140A': {'terminals': ('N. Platte', 'Maxwell'),
                            'atp_segs': ('L40AA', 'L40AB', 'L40AC'),
                            'atp_branches': (('NPLAT', 'NPMAX'),
                                             ('MAXWE', 'MAXNP'))},
                 'L1140B': {'terminals': ('Maxwell', 'Callaway'),
                            'atp_segs': ('L140B',),
                            'atp_branches': (('MAXWE', 'MAXCA'),
                                             ('CALLA', 'CAMAX'))},
                 'L1140C': {'terminals': ('Callaway', 'Broken Bow'),
                            'atp_segs': ('L140C',),
                            'atp_branches': (('CALLA', 'CALBB'),
                                             ('BROKE', 'BBCAL'))},
                 'L1240A': {'terminals': ('Broken Bow', 'BB Wind'),
                            'atp_segs': ('L40A1', 'L40A2', 'L40A3', 'L240B'),
                            'atp_branches': (('BROKE', 'BBMCK'),
                                             ('BBWIN', 'BBWMC'))},
                 'L1170':  {'terminals': ('Broken Bow', 'Loup City'),
                            'atp_segs': ('L170A', 'L170B'),
                            'atp_branches': (('BROKE', 'BBLC'),
                                             ('LOUPC', 'LCBB'))},
                 'L1074':  {'terminals': ('Broken Bow', 'Crooked Creek'),
                            'atp_segs': ('L1074',),
                            'atp_branches': (('BROKE', 'BBCCK'),
                                             ('CROOK', 'CCKBB'))},
                 'L1269':  {'terminals': ('Loup City', 'St. Libory'),
                            'atp_segs': ('L1269',),
                            'atp_branches': (('LOUPC', 'LCSTL'),
                                             ('STLIB', 'STLLC'))},
                 'L1188':  {'terminals': ('Loup City', 'N. Loup'),
                            'atp_segs': ('L1188',),
                            'atp_branches': (('LOUPC', 'LCNL'),
                                             ('NLOUP', 'NLLC'))},
                 'L1088':  {'terminals': ('N. Loup', 'Ord'),
                            'atp_segs': ('L1088',),
                            'atp_branches': (('NLOUP', 'NLORD'),
                                             ('ORD', 'ORDNL'))},
                 'L1192A': {'terminals': ('N. Loup', 'Spalding'),
                            'atp_segs': ('L192A',),
                            'atp_branches': (('NLOUP', 'NLSPA'),
                                             ('SPALD', 'SPANL'))},
                 'L1192B': {'terminals': ('Spalding', 'Albion'),
                            'atp_segs': ('L192B',),
                            'atp_branches': (('SPALD', 'SPAAL'),
                                             ('ALBIO', 'ALSPA'))}}

    # Save meter data to lookup dict keyed on location and terminal to make
    # it easy to access data. The last data found for any given location and
    # terminal will be saved. It would be possible to save all data, but that
    #  would add another layer to the data structure.
    met_lookup = {}
    for m in data_list:
        met_lookup[(m['Location'], m['Terminal'])] = m
    for line, l in line_defs.items():
        for idx, terminal in enumerate(l['terminals']):
            try:
                l['term%d_met' % idx] = met_lookup[(terminal, line)]
            except KeyError:
                pass

    # Pull in ATP parameters from PCH files
    for line, l in line_defs.items():
        _, summary_data_dict = pyATP.get_line_params_from_pch(
            args.atp_pch_folder, l['atp_segs']
        )
        # Add summary line parameter data to data dict.
        for k, v in summary_data_dict.items():
            l[k] = v

    # Pull in ATP steady-state results gathered by make_ss_csv
    for line, l in line_defs.items():
        for idx, branch in enumerate(l['atp_branches']):
            l['term%d_atp_branch' % idx] = atp_data_list['branch_data'][branch]
            l['term%d_atp_bus' % idx] = atp_data_list['bus_data'][branch[0]]

    # Calculate absolute bus angles based on a reference bus and line PQ flows.
    bus_angles = set_angles(line_defs, 'Thedford')
    print('\n'.join(('{:16}: {:.2f} deg.'.format(bus, bus_angles[bus])
                     for bus in sorted(bus_angles))))

    for line in sorted(line_defs):
        l = line_defs[line]
        T = l['ABCD']
        Z = l['Zeq']
        Z_s = l['Zeq_s']
        bus0 = l['terminals'][0]
        bus1 = l['terminals'][1]

        print('')
        print('=' * 80)
        print(line)
        # print(line_defs[line]['Zeq'])
        print('Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}'\
              .format(l['Zsum_s'][1, 1],
                      l['Zsum_s'][0, 0],
                      Polar(Z_s[2, 1])))

        try:
            X1_ph_m, = terminal_state(line_defs, line, terminal=(0,))
            have_met_1 = True
            # Shift measured phasors by calculated angle relative to ref. bus
            X1_ph_m *= cmath.exp(1j * math.radians(bus_angles[bus0]))
            # Calculate symmetrical components
            X1_s_m = lineZ.ph_to_seq_v(X1_ph_m)
        except KeyError:
            have_met_1 = False
            X1_ph_m = np.empty(6)
            X1_ph_m.fill(np.NAN)
            X1_s_m = np.empty(6)
            X1_s_m.fill(np.NAN)

        try:
            X2_ph_m, = terminal_state(line_defs, line, terminal=(1,))
            X2_ph_m *= cmath.exp(1j * math.radians(bus_angles[bus1]))
            X2_s_m = lineZ.ph_to_seq_v(X2_ph_m)
            have_met_2 = True
        except KeyError:
            have_met_2 = False
            X2_ph_m = np.empty(6)
            X2_ph_m.fill(np.NAN)
            X2_s_m = np.empty(6)
            X2_s_m.fill(np.NAN)

        X1_s_atp = np.concatenate((l['term0_atp_bus']['seq_voltages'],
                                   l['term0_atp_branch'][
                                       'seq_br_currents']))
        #
        X1_s_atp[3:] *= -1 # Reverse polarity of I1
        X2_s_atp = np.concatenate((l['term1_atp_bus']['seq_voltages'],
                                   l['term1_atp_branch'][
                                       'seq_br_currents']))

        print('-' * 80)

        with printoptions(precision=5, suppress=True,
                          formatter={'complex_kind': polar_formatter}):
            if False:
                if have_met_1:
                    print('%s Metering' % l['terminals'][0])
                    print('Voltages (V): ', X1_ph_m[:3])
                    print('Current  (A): ', X1_ph_m[3:])
                    print('Symmetrical Components')
                    print('Voltages (V): ', X1_s_m[:3])
                    print('Current  (A): ', X1_s_m[3:])

                print('From ATP model for Terminal %s' % l['terminals'][0])
                print('Voltages (V): ', l['term0_atp_bus']['ph_voltages'])
                print('Current  (A): ', l['term0_atp_branch']['ph_br_currents'])
                print('Symmetrical Components')
                print('Voltages (V): ', l['term0_atp_bus']['seq_voltages'])
                print('Current  (A): ', l['term0_atp_branch']['seq_br_currents'])

                print('-' * 80)

                if have_met_2:
                    print('%s Metering' % l['terminals'][1])

                    print('Voltages (V): ', X2_ph_m[:3])
                    print('Current  (A): ', X2_ph_m[3:])
                    print('Symmetrical Components from %s' % l['terminals'][1])
                    print('Voltages (V): ', X2_s_m[:3])
                    print('Current  (A): ', X2_s_m[3:])

                print('From ATP model for Terminal %s' % l['terminals'][1])
                print('Voltages (V): ', l['term1_atp_bus']['ph_voltages'])
                print('Current  (A): ', l['term1_atp_branch']['ph_br_currents'])
                print('Symmetrical Components')
                print('Voltages (V): ', l['term1_atp_bus']['seq_voltages'])
                print('Current  (A): ', l['term1_atp_branch']['seq_br_currents'])


                if have_met_1 and have_met_2:
                    print('-' * 80)
                    X1_ph = X1_ph_m
                    X2_ph = T.dot(X1_ph)

                    print('Calculated vs measured phase angles at %s'
                          % l['terminals'][1])
                    print('Phase       ', np.angle(X2_ph, deg=True)
                          - np.angle(X2_ph_m, deg=True))
                    print('Symm. Comp. ', np.angle(lineZ.ph_to_seq_v(X2_ph), deg=True)
                          - np.angle(X2_s_m, deg=True))

                print('-' * 80)

            print('')
            print('Comparison to ATP          Relay MET       ATP Steady-State'
                  '      Difference')
            print(' '*23 + '-'*17 + '  ' + '-'*17 + '  ' + '-'*17)
            print('{:<23}{:7.2f}  {:7.2f}  {:7.2f}'
                  .format(bus0 + ' Bus V2:',
                          Polar(X1_s_m[2]),
                          Polar(X1_s_atp[2]),
                          Polar(X1_s_atp[2] - X1_s_m[2])))
            print('{:<23}{:7.2f}  {:7.2f}  {:7.2f}'
                  .format(bus1 + ' Bus V2:',
                          Polar(X2_s_m[2]),
                          Polar(X2_s_atp[2]),
                          Polar(X2_s_atp[2] - X2_s_m[2])))

            vdrop2_m = X2_s_m[2] - X1_s_m[2]
            vdrop2_atp = X2_s_atp[2] - X1_s_atp[2]
            print('{:<23}{:7.2f}  {:7.2f}  {:7.2f}'
                  .format('Delta V2:',
                          Polar(vdrop2_m),
                          Polar(vdrop2_atp),
                          Polar(vdrop2_m - vdrop2_atp)))

            vdrop1_m = X2_s_m[1] - X1_s_m[1]
            vdrop1_atp = X2_s_atp[1] - X1_s_atp[1]
            print('{:<23}{:7.2f}  {:7.2f}  {:7.2f}'
                  .format('Delta V1:',
                          Polar(vdrop1_m),
                          Polar(vdrop1_atp),
                          Polar(vdrop1_m - vdrop1_atp)))

            if not have_met_1 and have_met_2:
                # If metering is only available at the other end, use it.
                # Switch both measured and ATP results for consistency.
                # Current at both ends should be very similar unless the line
                # is long or energized at EHV.
                X1_s_m = X2_s_m
                X1_s_m[3:] *= -1 # Invert current
                X1_s_atp = X2_s_atp
                X1_s_atp[3:] *= -1

            X1_s_diff =X1_s_atp - X1_s_m

            print('I1:                    {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(X1_s_m[4]),
                          Polar(X1_s_atp[4]),
                          Polar(X1_s_diff[4])))
            print('I1*Z1 Voltage Drop:    {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(X1_s_m[4] * Z_s[1, 1]),
                          Polar(X1_s_atp[4] * Z_s[1, 1]),
                          Polar(X1_s_diff[4] * Z_s[1, 1])))
            print('Z1 based on MET:       {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(vdrop1_m / X1_s_m[4]),
                          Polar(vdrop1_atp / X1_s_atp[4]),
                          vdrop1_atp / X1_s_atp[4]
                                - vdrop1_m / X1_s_m[4]))
            print('I2:                    {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(X1_s_m[5]),
                          Polar(X1_s_atp[5]),
                          Polar(X1_s_diff[5])))
            print('I2*Z2 Voltage Drop:    {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(X1_s_m[5] * Z_s[2, 2]),
                          Polar(X1_s_atp[5] * Z_s[2, 2]),
                          Polar(X1_s_diff[5] * Z_s[2, 2])))
            print('I1*Z21 Voltage Drop:   {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(X1_s_m[4] * Z_s[2, 1]),
                          Polar(X1_s_atp[4] * Z_s[2, 1]),
                          Polar(X1_s_diff[4] * Z_s[2, 1])))
            print('Total V2 Voltage Drop: {:7.2f}  {:7.2f}  {:7.2f}'
                  .format(Polar(X1_s_m[5] * Z_s[2, 2] + X1_s_m[4] * Z_s[2, 1]),
                          Polar(X1_s_atp[5] * Z_s[2, 2]
                                + X1_s_atp[4] * Z_s[2, 1]),
                          Polar(X1_s_diff[5] * Z_s[2, 2]
                                + X1_s_diff[4] * Z_s[2, 1])))
            print('Vdrop_2 - Z*I:         {:7.2f}  {:7.2f}'
                  .format(Polar(vdrop2_m - (X1_s_m[5] * Z_s[2, 2]
                                + X1_s_m[4] * Z_s[2, 1])),
                          Polar(vdrop2_atp - (X1_s_atp[5] * Z_s[2, 2]
                                + X1_s_atp[4] * Z_s[2, 1]))))

    print("--- Completed in %s seconds ---" % (time_fun() - start_time))


if __name__ == '__main__':
    sys.exit(main())
