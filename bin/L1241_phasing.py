#! /usr/bin/env python
# -- coding: utf-8 --

from __future__ import print_function, unicode_literals

import pyATP
import lineZ
from lineZ import Polar

import sys, shutil, glob, itertools, codecs, os, math
import numpy as np
np.set_printoptions(linewidth=120, precision=4)

# =============================================================================
# Configure file names and bus names

proj_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/'
atp_filenames = ['L1241Phasing2.atp', 'L1241Phasing3.atp']
model_weights = [1., 2.] # Weight no wind more than full wind.

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
def transitions(from_str, to_str, bot_to_center = True):
    ''' Returns a list of the possible transitions from from_str to to_str
        bot_to_center indicates whether the transition from delta to horizontal should allow
        the bottom conductor in the delta to move to the center position in the horizontal configuration.
        limit_SP indicates whether the transition from delta to vertical and back to delta should allow
        all phase change combinations or whether only the top two phases should be allowed to swap. Set
        to True to limit the transition to only swapping the top two phases.
        
        _ indicates a non-transposing continuation
        * indicates a transition that sets bot_to_center to true.
    '''
    if from_str[-1] == '_':
        # Non-transposing continuation
        return [(0, 1, 2)]
    if from_str[-1] == '*':
        bot_to_center = True
    from_str2 = from_str.rstrip('_*')
    to_str2 = to_str.rstrip('_*')
    if from_str2 == 'SP' and to_str2 == 'SP':
        # For this study, vertical structures are kept the same top-to-bottom as delta
        # If transition is allowed, it is just inverting positions of the top two
        # faces.
        return ((0, 1, 2), (0, 2, 1))
    if from_str2 == 'H' and to_str2 =='SP':
        return ((1, 0, 2), (2, 0, 1)) if bot_to_center else ((2, 0, 1),)

    if to_str2 == 'H' and from_str2 == 'SP':
        return ((1, 0, 2), (1, 2, 0)) if bot_to_center else ((1, 2, 0),)
    
    if from_str2 =='H' and to_str2 == 'H':
        # H-frame transposition structure lets either outside phase move to center
        return ((0,2,1), (1,0,2), (0,1,2))
    raise ValueError('from_str = '+from_str+', to_str = '+to_str)

# =============================================================================
# Divide data into some different arrays and print output to the screen
    
PIs = np.array([s[0] for s in sections])
PI_desc = [s[2] for s in sections]
section_ATPname = [s[3] for s in sections[:-1]]
# L is an array of the lengths of the segments
md_list = ['### Line Sections:']
for n, s_start, s_end in zip(range(1,len(sections)), sections[:-1], sections[1:]):
    md_list.append('Mile %.3f: %s' % (s_start[0], s_start[2]))
    md_list.append('    Section %d: %s (%.3f mi)' % (n, Str_names[s_start[1].rstrip('_*')], s_end[0] - s_start[0]))
md_list.append('Mile %.3f: %s' % (s_end[0], s_end[2]))
print('\n'.join(md_list))
L = PIs[1:] - PIs[:-1]

str_types = [s[1] for s in sections[:-1]]

transitions_list = [((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 1, 0),
                    (2, 0, 1))] + [transitions(from_str, to_str,
                                               bot_to_center=False)
                    for from_str, to_str in zip(str_types[:-1], str_types[1:])]

# Shorter list for debugging program
#transitions_list = [((0, 1, 2), (0, 2, 1))] + [transitions(from_str, to_str,
#                                               bot_to_center=False)
#                    for from_str, to_str in zip(str_types[:-1], str_types[1:])]

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
def max_neg_seq_unbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_unbalance) '''
    rtn = np.max(r[2])
    if not as_str:
        return rtn
    else:
        return 'Maximum neg. seq. voltage unbalance: %.4f %%' % rtn
max_neg_seq_unbalance.description = 'Max. Negative Seq. Unbalance'
max_neg_seq_unbalance.units = '%'
    
def avg_neg_seq_unbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_unbalance) '''
    rtn = np.mean(r[2])
    if not as_str:
        return rtn
    else:
        return 'Average neg. seq. voltage unbalance: %.4f %%' % rtn
avg_neg_seq_unbalance.description = 'Avg. Negative Seq. Unbalance'
avg_neg_seq_unbalance.units = '%'


def Z21_magnitude(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_unbalance,
        summary_line_data) '''
    if r[3] is None:
        return 0.
    Z21 = r[3]['Zsum_s'][2, 1]
    if not as_str:
        return abs(Z21)
    else:
        return 'Z21: {:.4f} Ohms'.format(Polar(Z21))
Z21_magnitude.description = 'Z21 magnitude'
Z21_magnitude.units = 'Ohms'


def Z_imbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_unbalance,
        summary_line_data) '''
    if r[3] is None:
        return 0.
    rtn = lineZ.impedance_imbalance(r[3]['Zsum'])
    if not as_str:
        return rtn
    else:
        Zph = np.absolute(r[3]['Zsum'].dot(lineZ.Apos))
        return 'Impedance imbalance: {} {:.4f} %'.format(Zph, rtn)
Z_imbalance.description = 'Impedance imbalance'
Z_imbalance.units = '%'

#criteria = [max_neg_seq_unbalance, avg_neg_seq_unbalance]
criteria = [max_neg_seq_unbalance,
            avg_neg_seq_unbalance,
            Z21_magnitude,
            Z_imbalance]
criteria_weights = [1., 1., 0., 10.] # Weight avg. more to scale it up.

# =============================================================================
# Without L1241 in model.
# ATP LIS file is saved from ATPDraw run and should not be over-written.

ph_voltages, seq_voltages, neg_seq_unbalance = pyATP.process_SS_bus_voltages(
    proj_dir + 'L1241Phasing_noL1241.lis', buses, RMS_scale = True)
r_noL1241 = ph_voltages, seq_voltages, neg_seq_unbalance, None
print('-'*80)
print('Without L1241 in the model:')
print('Phase voltages and negative-sequence unbalance voltage')
for n, b in enumerate(buses):
    print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
    
for c in criteria:
    print(c(r_noL1241, as_str=True))
    
# Hold results of each model in a dict indexed by the ATP model name
results_dict = lineZ.new_results_dict(all_transitions_list, atp_filenames)
r_base = lineZ.new_results_dict(['base'], atp_filenames)

for atp_filename in atp_filenames:
    # =============================================================================
    # Copy working files to temp directory
    
    # Main ATP model
    shutil.copyfile(proj_dir+atp_filename, tmp_dir+atp_filename)
    
    # Line segment data files
    for f in itertools.chain(glob.glob(proj_dir + '*.dat'),
                             glob.glob(proj_dir + '*.lib')):
        shutil.copy(f, tmp_dir)
    
    ATP_file = tmp_dir + atp_filename
    LIS_file = pyATP.lis_filename(ATP_file)
    
    # Find/replace directory name in ATP file so .lib files will be included properly.
    with codecs.open(ATP_file,'r', encoding='cp1252', errors='replace') as f:
        filedata = f.read()
    
    find = os.path.abspath(proj_dir)
    replace = os.path.abspath(tmp_dir)
    
    newdata = filedata.replace(find, replace)
    
    with codecs.open(ATP_file,'w', encoding='cp1252', errors='replace') as f:
        f.write(newdata)
        
    # =============================================================================
    # Run the model before any changes are made by the program as a sanity check.
    
    print('-'*80)
    print('With L1241 in the model, phased as in the model:')
    print('Phase voltages and negative-sequence unbalance voltage')
    pyATP.run_ATP(ATP_file)
    
    ph_voltages, seq_voltages, neg_seq_unbalance = \
        pyATP.process_SS_bus_voltages(LIS_file, buses, RMS_scale=True)

    # Read in line impedance parameters from PCH files
    seg_data_dict, summary_data_dict = pyATP.get_line_params_from_pch(
        tmp_dir, section_ATPname)

    r_base['base'][atp_filename] = ((ph_voltages,
                                     seq_voltages,
                                     neg_seq_unbalance,
                                     summary_data_dict,
                                     seg_data_dict),)
    
    for n, b in enumerate(buses):
        print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
        
    for c in criteria:
        print(c(r_base['base'][atp_filename][0], as_str=True))
    
    print('-'*80)
    
    # =============================================================================
    # Run analysis of all data cases. 
    # Time to run is approximately 1-2 s per case for this model.    
    results = []
    print('Running calcs....')
    for n, t in all_transitions_dict.items():
        for n2, l in enumerate(t):
            print('For %d transpositions, case %d of %d' % (n, n2, len(t)))
            
            # Set phasing of line sections in ATP .dat files & re-run line constants.
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
                
                # Make .lib file
                line_pch = tmp_dir + s + '.pch'
                with open(line_pch, 'r') as f:
                    pchlines = f.readlines()
                liblines = pyATP.make_ATPline_lib(pchlines)
                line_lib = tmp_dir + s + '.lib'
                with open(line_lib, 'w') as f:
                    f.writelines(liblines)
                
            # Run main ATP model with modified line sections
            pyATP.run_ATP(ATP_file)
            ph_voltages, seq_voltages, neg_seq_unbalance = \
                pyATP.process_SS_bus_voltages(LIS_file, buses, RMS_scale=True)

            # Read in line impedance parameters from PCH files
            seg_data_dict, summary_data_dict = \
                pyATP.get_line_params_from_pch(tmp_dir, section_ATPname)

            results_dict[l][atp_filename] = ((ph_voltages,
                                              seq_voltages,
                                              neg_seq_unbalance,
                                              summary_data_dict,
                                              seg_data_dict),)
    
# =============================================================================
# Filter to non-dominated results across all models.

filtered_results_dict = lineZ.filter_nondominated_results_multimodel(results_dict, criteria)

# Compute weighted criteria results
soln_list, weighted_results = lineZ.apply_criteria_weighting(results_dict, criteria, model_weights, criteria_weights)
weights_results_dict = {soln: wt for soln, wt in zip(soln_list, weighted_results)}
  
# =============================================================================
# Print results

# Base case
print('-'*80)
print('With L1241 in the model, phased as in the model:')
print('Phase voltages and negative-sequence unbalance voltage')
for model in atp_filenames:
    print('Model: %s' % model)
    ph_voltages, seq_voltages, neg_seq_unbalance, _, _ = \
        r_base['base'][model][0]

    for n, b in enumerate(buses):
        print('%6s : %s, %.6f' % (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)), neg_seq_unbalance[n]))
    
    for c in criteria:
        print(c(r_base['base'][model][0], as_str=True))
_, base_weighted_results = lineZ.apply_criteria_weighting(r_base, criteria, model_weights, criteria_weights)
print('Weighted results: %.4f' % base_weighted_results)

print('-'*80)
print('Results of best option(s)')
subtitle = 'Weights:'
subtitle += ', '.join([' %s: %.0f' % (m, m_wt) for m, m_wt in zip(atp_filenames, model_weights)])
subtitle += '\n'
subtitle +=  ', '.join(['%s: %.0f' % (f.description, f_wt) for f, f_wt in zip(criteria, criteria_weights)])
print(subtitle)
best_soln = {'best': None} 
#for soln, r in filtered_results_dict.items():
for soln, wt in sorted(zip(soln_list, weighted_results), key=lambda k: k[1]):
    r = results_dict[soln]
    if best_soln['best'] is None:
        best_soln['best'] = r
    print('-'*80)
    #ph_voltages, seq_voltages, neg_seq_unbalance = r[0]
    print(soln[0], soln[5])
    for model in atp_filenames:
        for c in criteria:
            print('%s, %s'% (model, c(r[model][0], as_str=True)))
    print('Weighted results: %.4f' % wt)

print('-'*80)
    
# Print dominated solutions. 
dominated_solns = [soln for soln in all_transitions_list
                   if soln not in filtered_results_dict ]
print('Dominated solutions:')
for soln in dominated_solns:
    print(soln[0], soln[5])
# =============================================================================
# Summarize selected solution
selected_soln = ((0, 1, 2), (1, 0, 2))
if selected_soln is not None:
    full_soln_tuple = tuple([s for s in results_dict
                             if (s[0], s[5]) == selected_soln][0])
    r = results_dict[full_soln_tuple]
    summary_data_dict = r[atp_filenames[0]][0][3]
    seg_data_dict = r[atp_filenames[0]][0][4]

    print('-'*80)
    print('Selected solution: ', selected_soln)

    for atp_filename in atp_filenames:
        r = results_dict[full_soln_tuple][atp_filename]
        ph_voltages = r[0][0]
        seq_voltages = r[0][1]
        neg_seq_unbalance = r[0][2]
        print('Bus p.u. voltages and % negative-sequence voltage')
        for n, b in enumerate(buses):
           print('%6s : %s, %.3f%%' %
                 (b, np.abs(ph_voltages[:,n].T)/(115e3/np.sqrt(3.)),
                  neg_seq_unbalance[n]))

    print('Total line impedance:')
    Z_s = summary_data_dict['Zsum_s']
    print('Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}' \
              .format(Z_s[1, 1],
                      Z_s[0, 0],
                      Polar(Z_s[2, 1])))
    phasing_info = lineZ.Pt_list_to_phasing(full_soln_tuple, str_types, Pos,
                                            Phase_list=('A', 'B', 'C'))
    print('Line Sections:')
    for n, phasing, s_start, s_end in zip(range(1,len(sections)), phasing_info,
                                          sections[:-1], sections[1:]):
        print(('Mile %.3f: %s' % (s_start[0], s_start[2]))
              + (' (Transposition)'
                 if n > 1
                    and sections[n-2][1].rstrip('_*')
                        == sections[n-1][1].rstrip('_*')
                    and full_soln_tuple[n-1]!=(0, 1, 2)
                 else ''))
        print('Section %d: %s (%.3f mi)' %
              (n, Str_names[s_start[1].rstrip('_*')],
               s_end[0] - s_start[0]))
        print(phasing)
        Z_s = lineZ.ph_to_seq_m(seg_data_dict[section_ATPname[n-1]].Z)
        print('    Z1 = {:.4f}, Z0 = {:.4f}, Z21 = {:.4f}' \
              .format(Z_s[1, 1],
                      Z_s[0, 0],
                      Polar(Z_s[2, 1])))
    print('Mile %.3f: %s' % (s_end[0], s_end[2]))