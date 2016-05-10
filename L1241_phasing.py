#! /usr/bin/python
# -- coding: utf-8 --

from __future__ import print_function, unicode_literals

import pyATP
import lineZ

import sys, shutil, glob, itertools, codecs, os
import numpy as np
np.set_printoptions(linewidth=120)

# =============================================================================
# Configure file names and bus names

proj_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/'
atp_filename = 'L1241Phasing.atp'

# Run all data in a temporary directory since the ATPDraw won't know
# to rebuild all the .pch files that this program monkeys with.
tmp_dir = 'C:/Users/pdbrown/Documents/ATPdata/work/L1241Phasing/tmp/'

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
# Enter line configuration information.

# Buses for monitoring voltage imbalance
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
# Without L1241 in model.
# ATP LIS file is saved from ATPDraw run and should not be over-written.

ph_voltages, seq_voltages, neg_seq_imbalance = pyATP.process_SS_bus_voltages(
                proj_dir + 'L1241Phasing_noL1241.lis', buses)

print('-'*80)
print('Without L1241 in the model:')
print('Phase voltages and negative-sequence imbalance voltage')
for n, b in enumerate(buses):
    print(b, ':', np.abs(ph_voltages[:,n].T)/(115e3*np.sqrt(2./3.)), neg_seq_imbalance[0,n])
    
print('Maximum neg. seq. voltage imbalance:', np.max(neg_seq_imbalance), '%')
print('Average neg. seq. voltage imbalance:', np.mean(neg_seq_imbalance), '%')

# =============================================================================
# Run the model before any changes are made by the program as a sanity check.

print('-'*80)
print('With L1241 in the model, phased as in the model:')
print('Phase voltages and negative-sequence imbalance voltage')
pyATP.run_ATP(ATP_file)

ph_voltages, seq_voltages, neg_seq_imbalance = pyATP.process_SS_bus_voltages(LIS_file, buses)

for n, b in enumerate(buses):
    print(b, ':', np.abs(ph_voltages[:,n].T)/(115e3*np.sqrt(2./3.)), neg_seq_imbalance[0,n])
    
print('Maximum neg. seq. voltage imbalance:', np.max(neg_seq_imbalance), '%')
print('Average neg. seq. voltage imbalance:', np.mean(neg_seq_imbalance), '%')

print('-'*80)

# =============================================================================
all_transitions_dict = lineZ.make_transitions_dict(transitions_list, str_types)
print('Number of phasing combinations by number of transpositions:')
for k, v in all_transitions_dict.items():
    print(k, len(v))

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
                inlines = f.readlines()
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
        ph_voltages, seq_voltages, neg_seq_imbalance = pyATP.process_SS_bus_voltages(LIS_file, buses)
        
        # Save results to array
        results.append((n, l, (ph_voltages, seq_voltages, neg_seq_imbalance)))

# =============================================================================
# Define evaluation criteria and filter to get the non-dominated results.        
def max_neg_seq_imbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_imbalance) '''
    rtn = np.max(r[2])
    if not as_str:
        return rtn
    else:
        return 'Maximum neg. seq. voltage imbalance: %.4f %%' % rtn
    
def avg_neg_seq_imbalance(r, as_str=False):
    ''' r is assumed to be (ph_voltages, seq_voltages, neg_seq_imbalance) '''
    rtn = np.mean(r[2])
    if not as_str:
        return rtn
    else:
        return 'Average neg. seq. voltage imbalance: %.4f %%' % rtn

criteria = [max_neg_seq_imbalance, avg_neg_seq_imbalance]

filtered_results = lineZ.filter_nondominated_results(results, criteria)

# =============================================================================
# Print results  
print('-'*80)
print('Results of best option(s)')      
for r in filtered_results:
    print('-'*80)
    ph_voltages, seq_voltages, neg_seq_imbalance = r[2]
    print(r[1])
    for c in criteria:
        print(c(r[2], as_str=True))
    #continue
    
    for n, b in enumerate(buses):
        print(b, ':', np.abs(ph_voltages[:,n].T)/(115e3*np.sqrt(2./3.)), neg_seq_imbalance[0,n])
    
    phasing_info = lineZ.Pt_list_to_phasing(r[1], str_types, Pos, Phase_list=('A', 'B', 'C'))
    print('Line Sections:')
    for n, phasing, s_start, s_end in zip(range(1,len(sections)), phasing_info, sections[:-1], sections[1:]):
        print(('Mile %.3f: %s' % (s_start[0], s_start[2])) + (' (Transposition)' if n > 1 and sections[n-2][1].rstrip('_*') == sections[n-1][1].rstrip('_*') and r[1][n-1]!=(0, 1, 2) else ''))
        print('Section %d: %s (%.3f mi)' % (n, Str_names[s_start[1].rstrip('_*')], s_end[0] - s_start[0]))
        print(phasing)
    print('Mile %.3f: %s' % (s_end[0], s_end[2]))

print('-'*80)


# =============================================================================
# Create plot if matplotlib is available
try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit()

xvals = np.array([avg_neg_seq_imbalance(r[2]) for r in results])
yvals = np.array([max_neg_seq_imbalance(r[2]) for r in results])

plt.scatter(xvals, yvals, c='b', s=80., label='All results')

xvals = np.array([avg_neg_seq_imbalance(r[2]) for r in filtered_results])
yvals = np.array([max_neg_seq_imbalance(r[2]) for r in filtered_results])

plt.scatter(xvals, yvals, c='r', s=80., label='Non-dominated solutions')

plt.xlabel('Avg. Negative Seq. Imbalance (%)')
plt.ylabel('Max. Negative Seq. Imbalance (%)')
plt.legend(scatterpoints=1, fontsize='small')
plt.show()









