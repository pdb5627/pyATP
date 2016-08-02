#! /usr/bin/python
'''
Module of utility functions to drive ATP from Python and extract results
At this time, only steady-state result extraction is supported.
'''

from __future__ import print_function, unicode_literals

from math import sqrt

import lineZ

try:
    import numpy as np
    from numpy.linalg import inv
except ImportError:
    pass
    
# To parse ATP files, the test_data_cards module is used.
# It is currently under development.
import text_data_cards as tdc

import itertools, copy, os

import subprocess, re, csv, codecs, shutil

ATP_path = 'C:/ATP/ATPmingw/'
ATP_exe = 'runATP_G.bat'

be_quiet = True

def run_ATP(ATP_file, quiet=None):
    kwargs = {}
    if quiet is not None and quiet or quiet is None and be_quiet:
        kwargs['stdout'] = open(os.devnull, 'w')
    rtn = subprocess.call((ATP_path+ATP_exe, ATP_file), **kwargs)
    
def atp_basename(ATP_file):
    '''
    Returns the base filename corresponding to the ATP filename provided.
    It basically just removes the .atp from the end.
    '''
    return re.match('(.*)\.atp$',ATP_file, flags=re.I).group(1)

def lis_filename(ATP_file):
    '''
    Returns the LIS filename corresponding to the ATP filename provided.
    This does not verify that the LIS file exists or that it is in the expected
    format.
    '''
    return atp_basename(ATP_file) + '.lis'
    
def replace_text(ATP_file, old_text, new_text, outfile=None, n=None):
    '''
    Replaces some text in the specified ATP file with new text. Since ATP uses
    fixed with fields in the file, new text will be left padded with spaces if
    new text is shorter than old text or truncated on the right if new text is
    longer. Optional parameter n indicates that ONE instance of the old text
    should be replaced, and the instance replaced will be the nth instance.
    Optional parameter outfile provides a filename for the modified file to be
    saved to. If outfile is not provided, the input file is overwritten.
    
    It is assumed that the text would appear only once on any given line.
    
    If the old text is not found, ValueError is raised and the file is not written.
    '''
    new_text_fixed = ' '*max(0, len(old_text)-len(new_text)) + \
                    new_text[:len(old_text)]
    
    i = 0
    with codecs.open(ATP_file, 'r', encoding='iso-8859-1', errors='replace') as f:
        infile = f.readlines()
    for ln, line in enumerate(infile):
        if old_text in line:
            i += 1
            if n is None or i == n:
                infile[ln] = line.replace(old_text, new_text_fixed)
                if n is not None:
                    break
                    
    if i == 0:
        raise ValueError('Text to replace not found')
    
    with codecs.open(outfile if outfile is not None else ATP_file, 'w',
            encoding='iso-8859-1', errors='replace') as f:
        f.write(''.join(infile))
    
def node_ph(node_name, ph):
        return node_name + (ph if node_name != "TERRA" else "")
        
def get_SS_results(LIS_file, RMS_scale=False):
    '''
    Extract steady-state results from LIS file. Results are returned as a 
    tuple in the following structure:
    (<Node Voltages>, <Branch Currents>)
    
    <Node Voltages> has the following structure:
    { <Node Name>: <Node Voltage> }
    Where <Node Name> is a string of the node name and <Node Voltage> is a
    complex number representing the phasor node voltage.
    
    <Branch Current> has the following structure:
    { (<From Node>, <To Node>): <Branch Current> }
    Where <From Node> and <To Node> are strings of the node name and
    <Branch Current> is a complex number representing the phasor branch current.
    
    By default the phasor values returned are NOT scaled from the ATP output to
    convert from peak values to RMS values. They can be scaled down by a factor
    of sqrt(2) by passing RMS_scale as True.
    
    TODO: Detect if ATP throws an error and raise an exception
    '''
    s = 1./sqrt(2) if RMS_scale else 1.0 # set scaling factor
    
    node_voltages = {}
    branch_currents = {}
    
    with open(LIS_file, 'r') as f:
        iter_input = f # No pre-processing needed, but it could be done here
        
        ss_res_start = re.compile('^Sinusoidal steady-state phasor solution, branch by branch')
        ss_node_end = re.compile('^ *Total network loss  P-loss  by summing injections')
        ss_sw_start = re.compile('^Output for steady-state phasor switch currents.')
        
        for line in iter_input:
            if ss_res_start.match(line):
                break
        else:
            print('Steady-state phasor solution not found.')
        
        # Skip next four lines to get to where results start
        for _ in range(4):
            line = next(iter_input)
        
        # Steady state phasor solution column definitions
        # Since the output is fixed width text, it is most reliable to parse it
        # using the width of the columns. The columns will be defined using starting
        # column number. Column numbers are set 1-based to match a text
        # editor. Two blank spaces between columns are assumed.
        col_nums = [2, 13, 23, 40, 61, 78, 99, 116, 133]
        c = [slice(start-1, end-3) for start, end in zip(col_nums[:-1], col_nums[1:])]
        
        # Get steady state node voltages and currents
        while ss_node_end.match(line) is None:
            # Line 1: Blank
            
            line = next(iter_input)
            if ss_node_end.match(line) is not None:
                break
            # Line 2
            from_node = line[c[0]].strip()
            from_v_re = float(line[c[2]])
            from_i_re = float(line[c[4]])
            
            line = next(iter_input)
            # Line 3
            from_v_im = float(line[c[2]])
            from_i_im = float(line[c[4]])
            
            line = next(iter_input)
            # Line 4: Blank
            
            line = next(iter_input)
            # Line 5
            to_node = line[c[1]].strip()
            to_v_re = float(line[c[2]])
            to_i_re = float(line[c[4]])
            
            line = next(iter_input)
            # Line 6
            to_v_im = float(line[c[2]])
            to_i_im = float(line[c[4]])
            
            node_voltages[from_node] = complex(from_v_re, from_v_im)*s
            node_voltages[to_node] = complex(to_v_re, to_v_im)*s
            branch_currents[(from_node,to_node)] = complex(from_i_re, from_i_im)*s
            branch_currents[(to_node,from_node)] = complex(to_i_re, to_i_im)*s

            line = next(iter_input)
            
        # Skip down to switch currents (if any)
        while not ss_sw_start.match(line):
            line = next(iter_input)
        
        # See if switch currents were found
        if ss_sw_start.match(line):
            # Eat the column header line
            line = next(iter_input)
            
            col_nums = [7, 17, 30, 48, 66, 85, 97, 115, 133]
            c = [slice(start-1, end-3) for start, end in zip(col_nums[:-1], col_nums[1:])]
            
            line = next(iter_input)
            while len(line[c[0]].strip()) > 0:
                from_node = line[c[0]].strip()
                to_node = line[c[1]].strip() if len(line[c[1]].strip()) > 0 else 'TERRA'
                if 'Open' in line[c[2]]:
                    from_i_re = 0.
                    from_i_im = 0.
                else:
                    from_i_re = float(line[c[2]])
                    from_i_im = float(line[c[3]])
                
                branch_currents[(from_node,to_node)] = complex(from_i_re, from_i_im)*s
                branch_currents[(to_node,from_node)] = complex(-1*from_i_re, -1*from_i_im)*s
                
                line = next(iter_input)
            
        
        return node_voltages, branch_currents

def output_ss_file(LIS_file, SS_file=None, buses=None, phases=('A', 'B', 'C'), RMS_scale=False):
    '''
    Extract steady-state phasor results from LIS file and output them to a _ss.csv
    file in comma-separated format.
    '''
    
    node_voltages, branch_currents = get_SS_results(LIS_file, RMS_scale)
    if buses is not None:
        ph_voltages, seq_voltages, neg_seq_imbalance = \
                process_SS_bus_voltages(LIS_file, buses, phases, RMS_scale)
    
    if SS_file is None:
        SS_file = re.match('(.*)\.lis$',LIS_file, flags=re.I).group(1) + '_ss.csv'
    with open(SS_file, 'wb') as csvfile:
        sswriter = csv.writer(csvfile)
        if buses is not None:
            # Output bus voltage results
            sswriter.writerow(list(itertools.chain(('Bus',),
                itertools.chain(*[('%s-phase Voltage (Real)' % ph,  
                  '%s-phase Voltage (Imag)' % ph) for ph in phases]),
                itertools.chain(*[('%s-sequence Voltage (Real)' % ph,  
                  '%s-sequence Voltage (Imag)' % ph) for ph in ('Zero', 'Positive', 'Negative')]),
                ('Neg. Seq. Unbalance Factor (%%)',))))
            for n, bus in enumerate(buses):
                sswriter.writerow(list(itertools.chain((bus,), 
                        itertools.chain(*zip(np.real(ph_voltages[:,n]),
                            np.imag(ph_voltages[:,n]))),
                        itertools.chain(*zip(np.real(seq_voltages[:,n]),
                            np.imag(seq_voltages[:,n]))),
                        (neg_seq_imbalance[n],))))
        
            sswriter.writerow(['--------']*13)
        
        
        # Output node voltage results
        sswriter.writerow(['Bus', 'Bus Voltage (Real)', 'Bus Voltage (Imag)'])
        for node, voltage in node_voltages.items():
            sswriter.writerow([node, voltage.real, voltage.imag])
            
        # Output file section separator
        sswriter.writerow([])
        sswriter.writerow(['--------']*4)
        sswriter.writerow([])
            
        # Output branch current results
        sswriter.writerow(['From Bus', 'To Bus', 'Branch Current (Real)', 'Branch Current (Imag)'])
        for nodes, current in branch_currents.items():
            sswriter.writerow([nodes[0], nodes[1], current.real, current.imag])
            
    return SS_file
            
def process_SS_bus_voltages(LIS_file, buses, phases=('A', 'B', 'C'), RMS_scale=False):
    ''' Parses LIS_file to get steady state results, then creates vectors of
        voltage phasors at the specified buses. Returns ph_voltages, 
        seq_voltages, neg_seq_imbalance as tuple of lists in same order as buses.
    '''
    node_voltages, branch_currents = get_SS_results(LIS_file, RMS_scale)
    
    ph_voltages = np.array([[node_voltages[b+p] for p in phases] for b in buses]).T
    seq_voltages = np.array(lineZ.ph_to_seq(ph_voltages))
    neg_seq_imbalance = np.abs(seq_voltages[2]/seq_voltages[1])*100
    
    return ph_voltages, seq_voltages, neg_seq_imbalance

def extract_ABCD(ATP_template, ATP_tmp, current_key, switch_key,
                 in_port, out_port,
                 test_current = 500., switch_close_t = '999.',
                 phases =  ('A', 'B', 'C')):


    V1_s = np.zeros((3,3), dtype=np.complex128)
    V1_o = np.zeros((3,3), dtype=np.complex128)
    I1_s = np.zeros((3,3), dtype=np.complex128)
    I1_o = np.zeros((3,3), dtype=np.complex128)
    V2_o = np.zeros((3,3), dtype=np.complex128)
    I2_s = np.zeros((3,3), dtype=np.complex128)

    for out_port_short in (True, False):
        for n, ph in enumerate(phases):
            shutil.copyfile(ATP_template, ATP_tmp)
            # Find/replace code numbers in template ATP file and copy to new file
            for n2, ph2 in enumerate(phases):
                replace_text(ATP_tmp, current_key,
                        ('%6f.' % (test_current if n2==n else test_current/1000.)),
                        n=n+1)
            replace_text(ATP_tmp, switch_key, '-1' if out_port_short else '1')

            # Run ATP on new file
            run_ATP(ATP_tmp)

            # Extract steady-state results
            LIS_results = lis_filename(ATP_tmp)
            node_voltages, branch_currents = get_SS_results(LIS_results)
            
            if out_port_short:
                for n2, ph2 in enumerate(phases):
                    V1_s[n2, n] = node_voltages[node_ph(in_port[0], ph2)]
                    I1_s[n2, n] = branch_currents[(node_ph(in_port[0], ph2),
                                        node_ph(in_port[1], ph2))]
                    I2_s[n2, n] = branch_currents[(node_ph(out_port[0], ph2),
                                        node_ph(out_port[1], ph2))] # removed +ph2 because 'TERRA'
            else:
                for n2, ph2 in enumerate(phases):
                    V1_o[n2, n] = node_voltages[node_ph(in_port[0], ph2)]
                    I1_o[n2, n] = branch_currents[(node_ph(in_port[0], ph2), node_ph(in_port[1], ph2))]
                    V2_o[n2, n] = node_voltages[node_ph(out_port[0], ph2)]

    A = V1_o.dot(inv(V2_o))
    B = V1_s.dot(inv(I2_s))
    C = I1_o.dot(inv(V2_o))
    D = I1_s.dot(inv(I2_s))

    ABCD = np.array(np.bmat([[A, B], [C, D]]))
    
    return ABCD

    
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

# Quick and dirty hack to build lib files. Will only work for three-phase lines.
ATPline_lib_head = ['''KARD  4  4  5  5  7  7
KARG  1  4  2  5  3  6
KBEG  3  9  3  9  3  9
KEND  8 14  8 14  8 14
KTEX  1  1  1  1  1  1
/BRANCH
''']

ATPline_lib_foot = ['''$EOF
ARG, IN___A, IN___B, IN___C, OUT__A, OUT__B, OUT__C
''']

def make_ATPline_lib(pch_file_lines):
    line_idx = 0
    # Find line where data starts. Assume we just have to skip comments 'C '
    while pch_file_lines[line_idx][:2] == 'C ':
        line_idx += 1
    card_lines = pch_file_lines[line_idx:]
    assert(len(card_lines) == 10) # To make sure the assumed simplified case is met.
    return ATPline_lib_head + card_lines + ATPline_lib_foot

def main():
    '''
    This module can be called from the command line. No functionality is
    implemented yet. In the future this could be used to call ATP and automatically
    export the steady-state results or run other analysis.
    '''
    import sys
    
    print(sys.argv)
    print('No functionality implemented at this time.')
            
if __name__ == "__main__":
    main()

















