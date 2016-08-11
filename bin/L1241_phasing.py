#! /usr/bin/env python
# -- coding: utf-8 --

from __future__ import print_function, unicode_literals

import pyATP
import lineZ
from lineZ import Polar

import shutil
import glob
import itertools
import numpy as np

np.set_printoptions(linewidth=120, precision=4)

# =============================================================================
# Configure file names and bus names

proj_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/'
models = ['']
model_weights = [1.]

# Run all data in a temporary directory since the ATPDraw won't know
# to rebuild all the .pch files that this program monkeys with.
tmp_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/tmp/'

# =============================================================================
# Enter line configuration information.

# Buses for monitoring voltage unbalance
buses = ['NPLAT', 'THEDF', 'STAPL', 'MAXWE', 'CALLA', 'LOUPC', 'NLOUP', 'BBWIN',
         'CROOK', 'ORD',   'SPALD', 'BROKE', 'MUDDY']
Pos = {}
Str_names = {}

# H-frame structures
Pos['H'] = ['EHS side', 'Center', 'OPGW side']
Str_names['H'] = 'H-frame'

# Single-pole structures
Pos['SP'] = ['Bot', 'Mid', 'Top']
Str_names['SP'] = 'Single-pole'

# Segment lengths & Configurations
sections = [(0,       'SP', 'Muddy Creek Sub',   'L241A'),
            ( 5.98,   'H',  'Transition',        'L241B'),
            ( 6.73,   'SP', 'Transition',        'L241C'),
            (11.69,   'H',  'Transition',        'L241D'),
            (16.16,   'SP*', 'Transition',       'L241E'),
            (23.27,   'H', 'Special Transition', 'L241F'),
            (30.01,   'SP', 'Transition',        'L241G'),
            (39.63,   '',   'Ord Sub', '')]


# =============================================================================
# Define transition possibilities for this study
def transitions(from_str, to_str, bot_to_center=True):
    """ Returns a list of the possible transitions from from_str to to_str
        bot_to_center indicates whether the transition from delta to horizontal
        should allow the bottom conductor in the delta to move to the center
        position in the horizontal configuration.

        limit_SP indicates whether the transition from delta to vertical and
        back to delta should allow all phase change combinations or whether
        only the top two phases should be allowed to swap. Set to True to
        limit the transition to only swapping the top two phases.
        
        _ indicates a non-transposing continuation
        * indicates a transition that sets bot_to_center to true.
    """
    if from_str[-1] == '_':
        # Non-transposing continuation
        return [(0, 1, 2)]
    if from_str[-1] == '*':
        bot_to_center = True
    from_str2 = from_str.rstrip('_*')
    to_str2 = to_str.rstrip('_*')
    if from_str2 == 'SP' and to_str2 == 'SP':
        # For this study, vertical structures are kept the same top-to-bottom
        # as delta. If transition is allowed, it is just inverting positions
        # of the top two phases.
        return (0, 1, 2), (0, 2, 1)
    if from_str2 == 'H' and to_str2 == 'SP':
        return ((1, 0, 2), (2, 0, 1)) if bot_to_center else ((2, 0, 1),)

    if to_str2 == 'H' and from_str2 == 'SP':
        return ((1, 0, 2), (1, 2, 0)) if bot_to_center else ((1, 2, 0),)
    
    if from_str2 == 'H' and to_str2 == 'H':
        # H-frame transposition structure lets either outside phase move to
        # center
        return (0, 2, 1), (1, 0, 2), (0, 1, 2)
    raise ValueError('from_str = '+from_str+', to_str = '+to_str)

# =============================================================================
# Divide data into some different arrays and print output to the screen
    
PIs = np.array([s[0] for s in sections])
PI_desc = [s[2] for s in sections]
section_ATPname = [s[3] for s in sections[:-1]]
# L is an array of the lengths of the segments
md_list = ['### Line Sections:']
for n, s_start, s_end in zip(range(1,len(sections)),
                             sections[:-1],
                             sections[1:]):
    md_list.append('Mile %.3f: %s' % (s_start[0], s_start[2]))
    md_list.append('    Section %d: %s (%.3f mi)' %
                   (n, Str_names[s_start[1].rstrip('_*')],
                    s_end[0] - s_start[0]))
md_list.append('Mile %.3f: %s' % (s_end[0], s_end[2]))
print('\n'.join(md_list))
L = PIs[1:] - PIs[:-1]

str_types = [s[1] for s in sections[:-1]]

transitions_list = [((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 1, 0),
                    (2, 0, 1))] + [transitions(from_str, to_str,
                                               bot_to_center=False)
                    for from_str, to_str in zip(str_types[:-1], str_types[1:])]

# =============================================================================
# Generate list of all possible transitions
all_transitions_dict = lineZ.make_transitions_dict(transitions_list, str_types)
print('Number of phasing combinations by number of transpositions:')

all_transitions_list = []
for k, v in all_transitions_dict.items():
    print(k, len(v))
    all_transitions_list.extend(v)
                     
# =============================================================================
# Define evaluation criteria and filter to get the non-dominated results.        
L1240A_segs = ['L40A1', 'L40A2', 'L40A3']
_, L1240A_params = pyATP.get_line_params_from_pch(
    proj_dir, L1240A_segs)
L1240A_Z21 = L1240A_params['Zsum_s'][2, 1]

L1240B_segs = ['L240B']
_, L1240B_params = pyATP.get_line_params_from_pch(
    proj_dir, L1240B_segs)
L1240B_Z21 = L1240B_params['Zsum_s'][2, 1]


def Z21_magnitude(r, as_str=False):
    """
    :param r: (summary_line_data, seg_data)
    :param as_str:
    :return:
    """
    if r[0] is None:
        return 0.
    Z21 = r[0]['Zsum_s'][2, 1]
    if not as_str:
        return abs(Z21)
    else:
        return '\n'.join(['Z21: {:.4f} Ohms'.format(Polar(Z21)),
                          'Z21: {:.4f} Ohms series w/ L1240A'
                          .format(Polar(Z21 + L1240A_Z21))])
Z21_magnitude.description = 'Z21 magnitude'
Z21_magnitude.units = 'Ohms'


def Z_imbalance(r, as_str=False):
    """
    :param r: (summary_line_data, seg_data)
    :param as_str:
    :return:
    """
    if r[0] is None:
        return 0.
    rtn = lineZ.impedance_imbalance(r[0]['Zsum'])
    if not as_str:
        return rtn
    else:
        Zph = lineZ.phase_impedances(r[0]['Zsum'])
        return 'Impedance imbalance: {} {:.4f} %'.format(Zph, rtn)
Z_imbalance.description = 'Impedance imbalance'
Z_imbalance.units = '%'

criteria = [Z21_magnitude, Z_imbalance]
criteria_weights = [1., 1.]


# Hold results of each model in a dict indexed by the ATP model name
results_dict = lineZ.new_results_dict(all_transitions_list, models)

for model in models:
    # =============================================================================
    # Copy working files to temp directory
    # Line segment data files
    for f in itertools.chain(glob.glob(proj_dir + '*.dat'),
                             glob.glob(proj_dir + '*.lib')):
        shutil.copy(f, tmp_dir)

    # =============================================================================
    # Run analysis of all data cases. 
    # Time to run is approximately 1-2 s per case for this model.    
    results = []
    print('Running calcs....')
    for n, t in all_transitions_dict.items():
        for n2, l in enumerate(t):
            print('For %d transpositions, case %d of %d' % (n, n2, len(t)))
            
            # Set phasing of line sections in ATP .dat files & re-run line
            # constants.
            for Pt, s in zip(lineZ.cum_Pt(l), section_ATPname):
                line_const = tmp_dir + s + '.dat'
                
                with open(line_const, 'r') as f:
                    inlines = f.read().splitlines()
                line_data = pyATP.LineConstCards()
                line_data.read(inlines)
                
                # Modify the phasing in .dat file.
                for idx, ph in enumerate(Pt):
                    line_data.data['conductors'][idx]['IP'] = ph + 1
                
                outlines = line_data.write()
                
                with open(line_const, 'w') as f:
                    f.writelines(outlines)
                # Run ATP on the .dat file to create .pch file.
                pyATP.run_ATP(line_const)

            # Read in line impedance parameters from PCH files
            seg_data_dict, summary_data_dict = \
                pyATP.get_line_params_from_pch(tmp_dir, section_ATPname)

            results_dict[l][model] = ((summary_data_dict,
                                       seg_data_dict),)
    
# =============================================================================
# Filter to non-dominated results across all models.

filtered_results_dict = \
    lineZ.filter_nondominated_results_multimodel(results_dict, criteria)

# Compute weighted criteria results
nondom_only = True
soln_list, weighted_results = lineZ.apply_criteria_weighting(
    filtered_results_dict if nondom_only else results_dict,
    criteria, model_weights, criteria_weights)
weights_results_dict = {soln: wt for soln, wt in
                        zip(soln_list, weighted_results)}
  
# =============================================================================
# Print results

print('-'*80)
print('Results of best option(s)')
subtitle = 'Weights:\n'
if len(models) > 1:
    subtitle += ', '.join([' %s: %.0f' % (m, m_wt)
                           for m, m_wt in zip(models, model_weights)])
    subtitle += '\n'
subtitle += ', '.join(['%s: %.0f' % (f.description, f_wt)
                       for f, f_wt in zip(criteria, criteria_weights)])
print(subtitle)

for soln, wt in sorted(zip(soln_list, weighted_results), key=lambda k: k[1]):
    r = results_dict[soln]
    print('-'*80)

    print('Solution key:', soln[0], soln[5])
    for model in models:
        if len(models) > 1:
            print('Model:', model)
        for c in criteria:
            print('%s' % c(r[model][0], as_str=True))
    print('Weighted evaluation: %.4f' % wt)

print('-'*80)
    
# Print dominated solutions. 
dominated_solns = [soln for soln in all_transitions_list
                   if soln not in filtered_results_dict]
print('Dominated solutions:')
for soln in dominated_solns:
    print(soln[0], soln[5])
# =============================================================================
# Summarize selected solution
# selected_soln = ((1, 2, 0), (1, 0, 2))  # Option A
selected_soln = ((0, 1, 2), (1, 0, 2))  # Option B
if selected_soln is not None:
    full_soln_tuple = tuple([s for s in results_dict
                             if (s[0], s[5]) == selected_soln][0])
    r = results_dict[full_soln_tuple]
    summary_data_dict = r[models[0]][0][0]
    seg_data_dict = r[models[0]][0][1]

    print('-'*80)
    print('Selected solution: ', selected_soln)

    print('Total line impedance:')
    Z_s = summary_data_dict['Zsum_s']
    print('Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}'
          .format(Z_s[1, 1],
                  Z_s[0, 0],
                  Polar(Z_s[2, 1])))

    for model in models:
        if len(models) > 1:
            print('Model:', model)
        for c in criteria:
            print('%s' % c(r[model][0], as_str=True))

    phasing_info = lineZ.Pt_list_to_phasing(full_soln_tuple, str_types, Pos,
                                            Phase_list=('A', 'B', 'C'))
    print('Line Sections:')
    for n, phasing, s_start, s_end in zip(range(1, len(sections)),
                                          phasing_info,
                                          sections[:-1], sections[1:]):
        print(('Mile %.3f: %s' % (s_start[0], s_start[2]))
              + (' (Transposition)'
                 if n > 1
                    and sections[n-2][1].rstrip('_*')
                        == sections[n-1][1].rstrip('_*')
                    and full_soln_tuple[n-1] != (0, 1, 2)
                 else ''))
        print('Section %d: %s (%.3f mi)' %
              (n, Str_names[s_start[1].rstrip('_*')],
               s_end[0] - s_start[0]))
        print(phasing)
        Z = seg_data_dict[section_ATPname[n - 1]].Z
        Z_s = lineZ.ph_to_seq_m(Z)
        print('    Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}'
              .format(Z_s[1, 1],
                      Z_s[0, 0],
                      Polar(Z_s[2, 1])))
        print('    Phase impedances & imbalance: {}, {:.2f}%'.format(
            lineZ.phase_impedances(Z),
            lineZ.impedance_imbalance(Z)))
    print('Mile %.3f: %s' % (s_end[0], s_end[2]))
