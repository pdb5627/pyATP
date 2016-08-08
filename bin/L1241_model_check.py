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


def terminal_states(line_defs, line):
    """ Extract phase voltages and currents from meter data in the line_defs
        dict.
    """
    rtn = []
    for s in ('term0_met', 'term1_met'):
        t = line_defs[line][s]
        V_ph = np.array(
            [t['V%s' % ph] for ph in ['A', 'B', 'C']]) * 1e3  # kV to V
        I_ph = np.array(
            [t['I%s' % ph] for ph in ['A', 'B', 'C']])  # already A
        # Reverse polarity of I1
        if s == 'term0_met':
            I_ph *= -1
        X_ph = np.concatenate((V_ph, I_ph))
        rtn.append(X_ph)
    return tuple(rtn)


def set_angles(line_defs, bus, buses_set=None):
    # On first call, set the bus to be reference
    if buses_set is None:
        buses_set = {bus: 14.4}

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
        #if 'V1_MAG' in l['term%d_met' % term]:
        try:
            V1_mag = 1000. * l['term%d_met' % term]['V1_MAG']
        except KeyError:
            try:
                #elif 'VA_MAG' in l['term%d_met' % term]:
                V1_mag = 1000. * l['term%d_met' % term]['VA_MAG']
            except KeyError:
                #else:
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

    line_defs = {'L1078B': {'terminals': ('Thedford', 'Stapleton'),
                            'atp_segs': ('L78BB', 'L78BA')},
                 'L1078A': {'terminals': ('Stapleton', 'Maxwell'),
                            'atp_segs': ('L78AB', 'L78AA')},
                 'L1140A': {'terminals': ('N. Platte', 'Maxwell'),
                            'atp_segs': ('L40AA', 'L40AB', 'L40AC')},
                 'L1140B': {'terminals': ('Maxwell', 'Callaway'),
                            'atp_segs': ('L140B',)},
                 'L1140C': {'terminals': ('Callaway', 'Broken Bow'),
                            'atp_segs': ('L140C',)},
                 'L1240A': {'terminals': ('Broken Bow', 'BB Wind'),
                            'atp_segs': ('L40A1', 'L40A2', 'L40A3', 'L240B')},
                 'L1170':  {'terminals': ('Broken Bow', 'Loup City'),
                            'atp_segs': ('L170A', 'L170B')},
                 'L1074':  {'terminals': ('Broken Bow', 'Crooked Creek'),
                            'atp_segs': ('L1074',)},
                 'L1269':  {'terminals': ('Loup City', 'St. Libory'),
                            'atp_segs': ('L1269',)},
                 'L1188':  {'terminals': ('Loup City', 'N. Loup'),
                            'atp_segs': ('L1188',)},
                 'L1088':  {'terminals': ('N. Loup', 'Ord'),
                            'atp_segs': ('L1088',)},
                 'L1192A': {'terminals': ('N. Loup', 'Spalding'),
                            'atp_segs': ('L192A',)},
                 'L1192B': {'terminals': ('Spalding', 'Albion'),
                            'atp_segs': ('L192B',)}}

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
    for line in line_defs:
        line_defs[line]['atp_params'] = []
        for seg in line_defs[line]['atp_segs']:
            with open(os.path.join(args.atp_pch_folder, seg + '.pch')) as \
                    pch_file:
                pch_lines = pch_file.readlines()
                params = pyATP.LineConstPCHCards()
                params.read(pch_lines)
                line_defs[line]['atp_params'].append(params)

    # Combine segment ABCD matrices into ABCD of line in phase and sequence
    # components
    for line, l in line_defs.items():
        l['Zsum'] = np.sum([pch_params.Z for pch_params in l['atp_params']], 0)
        l['Ysum'] = np.sum([pch_params.Y for pch_params in l['atp_params']], 0)
        l['Zsum_s'] = lineZ.ph_to_seq_m(l['Zsum'])
        l['Ysum_s'] = lineZ.ph_to_seq_m(l['Ysum'])
        ABCD_list = [pch_params.ABCD for pch_params in l['atp_params']]
        l['ABCD'] = lineZ.combine_ABCD(ABCD_list)
        Z, Y1, Y2 = lineZ.ABCD_to_ZY(l['ABCD'])
        l['Zeq'] = Z
        l['Yeq'] = Y1 + Y2
        l['ABCD_s'] = lineZ.ph_to_seq_m(l['ABCD'])
        Z, Y1, Y2 = lineZ.ABCD_to_ZY(l['ABCD_s'])
        l['Zeq_s'] = Z
        l['Yeq_s'] = Y1 + Y2

    # Calculate absolute bus angles based on a reference bus and line PQ flows.
    bus_angles = set_angles(line_defs, 'Thedford')
    print('\n'.join(('{:16}: {:.2f} deg.'.format(bus, a) for bus,
                                                      a in bus_angles.items())))

    for line, l in line_defs.items():
        T = l['ABCD']
        Z = l['Zeq']
        Z_s = l['Zeq_s']
        bus0 = l['terminals'][0]
        bus1 = l['terminals'][1]


        print('-'*80)
        print(line)
        # print(line_defs[line]['Zeq'])
        print('Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}'\
              .format(l['Zsum_s'][1, 1],
                      l['Zsum_s'][0, 0],
                      Polar(Z_s[2, 1])))

        # for p in itertools.permutations((0, 1, 2)):
        #     print('{}, {:.4f}'.format(p, Polar(lineZ.ph_to_seq_m(\
        #         lineZ.apply_phasing(Z, p))[2, 1])))

        # plot_data = lineZ.ph_to_seq_m(T) - np.eye(6)
        # print(plot_data)
        # column_labels = ['V1_A', 'V1_B', 'V1_C', 'I1_A', 'I1_B', 'I1_C']
        # row_labels = ['V2_A', 'V2_B', 'V2_C', 'I2_A', 'I2_B', 'I2_C']
        # fig, ax = plt.subplots()
        # heatmap = ax.pcolor(plot_data, cmap=plt.cm.Blues)
        # ax.set_xticks(np.arange(plot_data.shape[0]) + 0.5, minor=False)
        # ax.set_yticks(np.arange(plot_data.shape[1]) + 0.5, minor=False)
        # ax.invert_yaxis()
        # ax.xaxis.tick_top()
        # ax.set_xticklabels(row_labels, minor=False)
        # ax.set_yticklabels(column_labels, minor=False)
        # plt.show()

        # pf = 0.1 # Pos-sequence power factor
        # S1_out = np.linspace(0., 120.e6)
        # V1 = np.broadcast_to(np.array([[1.], [68.5e3], [350.]]), (3, 50))
        # S2_out = np.broadcast_to(3500.j, S1_out.shape)
        # S0_out = np.broadcast_to(0., S1_out.shape)
        # S_out = np.stack((S0_out, S1_out*(pf - 1j*np.sqrt(1 - pf**2)), S2_out))
        # I1 = np.conj(S_out/V1/3)
        # X1 = np.concatenate((V1, I1), axis=0)
        # X2 = T_s.dot(X1)
        # V2 = X2[:3, :]
        # I2 = X2[3:, :]
        # S_in = 3*V2*np.conj(I2)
        # delta_S = S_in - S_out
        # delta_P = delta_S.real
        # delta_Q = delta_S.imag
        # delta_V = np.absolute(V2) - np.absolute(V1)
        # plt.subplot(2, 1, 1)
        # plt.title('Positive-sequence Line Losses')
        # plt.plot(S1_out / 1e6, delta_P[1, :]/1e6, label='P')
        # plt.plot(S1_out / 1e6, delta_Q[1, :]/1e6, label='Q')
        # plt.plot(S1_out / 1e6, delta_V[1, :] / 1e3, label='V')
        # plt.legend()
        # plt.subplot(2, 1, 2)
        # plt.title('Negative-sequence Line Losses')
        # plt.plot(S1_out / 1e6, delta_P[2, :], label='P')
        # plt.plot(S1_out / 1e6, delta_Q[2, :], label='Q')
        # plt.plot(S1_out / 1e6, delta_V[2, :], label='V')
        # plt.legend()
        # plt.show()
        # print(S_in - S_out)
        # break

        try:
            X1_ph_m, X2_ph_m = terminal_states(line_defs, line)
        except KeyError:
            continue
        print(line)

        # Shift measured phasors by calculated angle relative to ref. bus
        X1_ph_m *= cmath.exp(1j*math.radians(bus_angles[bus0]))
        X2_ph_m *= cmath.exp(1j * math.radians(bus_angles[bus1]))

        # Calculate symmetrical components
        X1_s_m = lineZ.ph_to_seq_v(X1_ph_m)
        X2_s_m = lineZ.ph_to_seq_v(X2_ph_m)

        with printoptions(precision=5, suppress=True,
                          formatter={'complex_kind': polar_formatter}):
            print('%s Metering' % l['terminals'][0])
            print('Voltages (V): ', X1_ph_m[:3])
            print('Current  (A): ', X1_ph_m[3:])
            print('Symmetrical Components')
            print('Voltages (V): ', X1_s_m[:3])
            print('Current  (A): ', X1_s_m[3:])

            print('%s Metering' % l['terminals'][1])

            print('Voltages (V): ', X2_ph_m[:3])
            print('Current  (A): ', X2_ph_m[3:])
            print('Symmetrical Components from %s' % l['terminals'][1])
            print('Voltages (V): ', X2_s_m[:3])
            print('Current  (A): ', X2_s_m[3:])

            X1_ph = X1_ph_m
            X2_ph = T.dot(X1_ph)
            print('Calculated based on ABCD matrix and Terminal 1 V & I:')
            print('Voltages (V): ', X2_ph[:3])
            print('Current  (A): ', X2_ph[3:])

            # print()
            print('Calculated vs measured phase angles')
            print('Phase ', np.angle(X2_ph, deg=True) - np.angle(X2_ph_m, deg=True))
            print('Symm. Comp. ', np.angle(lineZ.ph_to_seq_v(X2_ph), deg=True)
                  - np.angle(X2_s_m, deg=True))

            print('I2*Z2 Voltage Drop:    {:7.2f}'
                  .format(Polar(X1_s_m[5] * Z_s[2, 2])))
            print('I1*Z21 Voltage Drop:   {:7.2f}'
                  .format(Polar(X1_s_m[4] * Z_s[2, 1])))
            print('Total V2 Voltage Drop: {:7.2f}'
                  .format(Polar(X1_s_m[5] * Z_s[2, 2] + X1_s_m[4] * Z_s[2, 1])))


        # print()
        # X1_seq = lineZ.ph_to_seq_v(X1_ph)
        # T_seq = lineZ.ph_to_seq_m(T)
        # X2_seq = T_seq.dot(X1_seq)
        # print(X1_seq)
        # print(X2_seq)
        # print(np.angle(X1_seq, deg=True))
        # print(np.angle(X2_seq, deg=True))
        # print(np.angle(X2_seq, deg=True) - np.angle(X1_seq, deg=True))

    print("--- Completed in %s seconds ---" % (time_fun() - start_time))


if __name__ == '__main__':
    sys.exit(main())
