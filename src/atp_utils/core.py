'''
Module of utility functions to drive ATP from Python and extract results
At this time, only steady-state result extraction is supported.
'''

from math import sqrt
import pickle

from atp_utils import sequence_components as lineZ

import numpy as np
from numpy.linalg import inv

import itertools, copy, os
from pathlib import Path
import tempfile
import functools

import subprocess, re, csv, codecs, shutil

from .dtypes import fdtype, cdtype
from .line_const_parsing import LineConstPCHCards


@functools.cache
def filesystem_is_case_sensitive():
    with tempfile.NamedTemporaryFile(prefix='TmP') as tmp_file:
        return(not os.path.exists(tmp_file.name.lower()))


ATP_path = Path('~/atplinux').expanduser()
ATP_exe = 'run_tpbig.sh'

be_quiet = True

def run_ATP(ATP_file, quiet=None):
    kwargs = {}
    if quiet is not None and quiet or quiet is None and be_quiet:
        kwargs['stdout'] = open(os.devnull, 'w')
    # Problems on Linux, due to case-sensitive filenames:
    # - ATP makes the filename all caps, which can result in opening a DIFFERENT file
    #   Solution: Only use an ALL CAPS filename. Error otherwise.
    ATP_file = Path(ATP_file)
    if not ATP_file.exists:
        raise OSError(f"Specified ATP file '{ATP_file}' does not exist.")

    ATP_fdir, ATP_fname = ATP_file.parent, ATP_file.name

    if filesystem_is_case_sensitive() and str(ATP_fname).upper() != str(ATP_fname):
        raise RuntimeError(f"Specified ATP file '{ATP_fname}' is not all caps.")

    # Change to directory of ATP file
    # This allows paths to files referenced by INCLUDE to be resolved relative to the ATP file
    # It also means tmp files made by ATP will also get dropped there.
    kwargs["cwd"] = ATP_fdir

    rtn = subprocess.call((os.path.join(ATP_path, ATP_exe), ATP_fname), **kwargs)

def atp_basename(ATP_file):
    '''
    Returns the base filename corresponding to the ATP filename provided.
    It basically just removes the .atp from the end.
    '''
    ATP_file = Path(ATP_file)
    return ATP_file.with_suffix('')

def lis_filename(ATP_file):
    '''
    Returns the LIS filename corresponding to the ATP filename provided.
    This does not verify that the LIS file exists or that it is in the expected
    format.
    '''
    # It seems that at least the atplinux version writes to a *lowercase* filename
    ATP_file = Path(ATP_file)
    return ATP_file.with_stem(ATP_file.stem.lower()).with_suffix(".lis")

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


def output_ss_file(LIS_file, SS_file=None, pickle_file=None,
                   buses=None, branches=None,
                   phases=('A','B', 'C'), RMS_scale=False):
    '''
    Extract steady-state phasor results from LIS file and output them to a _ss.csv
    file in comma-separated format.
    '''
    data_list = {} # Data to save to pickle file
    node_voltages, branch_currents = get_SS_results(LIS_file, RMS_scale)
    data_list['node_voltages'] = node_voltages
    data_list['branch_currents'] = branch_currents

    if buses is not None:
        ph_voltages, seq_voltages, neg_seq_imbalance = \
                process_SS_bus_voltages(LIS_file, buses, phases, RMS_scale)
        bus_data = {}
        for n, bus in enumerate(buses):
            bus_data[bus] = {}
            bus_data[bus]['ph_voltages'] = ph_voltages[:, n]
            bus_data[bus]['seq_voltages'] = seq_voltages[:, n]
            bus_data[bus]['neg_seq_imbalance'] = neg_seq_imbalance[n]
        data_list['bus_data'] = bus_data

    if branches is not None:
        ph_br_currents, seq_br_currents, S_3ph = \
            process_SS_branch_currents(LIS_file, branches, phases, RMS_scale)
        branch_data = {}
        for n, branch in enumerate(branches):
            branch = tuple(branch) # convert to tuple for indexing
            branch_data[branch] = {}
            branch_data[branch]['ph_br_currents'] = ph_br_currents[:, n]
            branch_data[branch]['seq_br_currents'] = seq_br_currents[:, n]
            branch_data[branch]['S_3ph'] = S_3ph[n]
        data_list['branch_data'] = branch_data

    # Reorganize data_list to be easier to key into bus or branch data
    bus_data = {}

    if SS_file is None:
        SS_file = re.match(r'(.*)\.lis$',LIS_file, flags=re.I).group(1) + '_ss.csv'

    if pickle_file is None:
        pickle_file = re.match(r'(.*)\.lis$',LIS_file, flags=re.I).group(1) + '_ss.p'

    with open(pickle_file, 'wb') as binfile:
        pickle.dump(data_list, binfile)

    with open(SS_file, 'w') as csvfile:
        sswriter = csv.writer(csvfile, lineterminator='\n')
        if buses is not None:
            # Output bus voltage results
            sswriter.writerow(list(itertools.chain(('Bus',),
                itertools.chain(*[('%s-phase Voltage (Real)' % ph,
                  '%s-phase Voltage (Imag)' % ph) for ph in phases]),
                itertools.chain(*[('%s-sequence Voltage (Real)' % ph,
                  '%s-sequence Voltage (Imag)' % ph) for ph in ('Zero', 'Positive', 'Negative')]),
                itertools.chain(*[('%s-phase Voltage (Mag)' % ph,
                  '%s-phase Voltage (Ang)' % ph) for ph in phases]),
                itertools.chain(*[('%s-sequence Voltage (Mag)' % ph,
                  '%s-sequence Voltage (Ang)' % ph) for ph in ('Zero', 'Positive', 'Negative')]),
                ('Neg. Seq. Unbalance Factor (%%)',))))
            for n, bus in enumerate(buses):
                sswriter.writerow(list(itertools.chain((bus,),
                        itertools.chain(*zip(np.real(ph_voltages[:, n]),
                            np.imag(ph_voltages[:,n]))),
                        itertools.chain(*zip(np.real(seq_voltages[:, n]),
                            np.imag(seq_voltages[:,n]))),
                        itertools.chain(*zip(np.absolute(ph_voltages[:, n]),
                           np.angle(ph_voltages[:, n], deg=True))),
                        itertools.chain(*zip(np.absolute(seq_voltages[:, n]),
                           np.angle(seq_voltages[:, n], deg=True))),
                        (neg_seq_imbalance[n],))))

            sswriter.writerow(['--------']*26)

        if branches is not None:
            # Output branch current results
            sswriter.writerow(list(itertools.chain(('From Bus', 'To Bus',
                                                    '3PH MW', '3PH Mvar'),
                itertools.chain(*[('%s-phase Voltage (Real)' % ph,
                  '%s-phase Voltage (Imag)' % ph) for ph in phases]),
                itertools.chain(*[('%s-sequence Voltage (Real)' % ph,
                  '%s-sequence Voltage (Imag)' % ph) for ph in ('Zero', 'Positive', 'Negative')]),
                itertools.chain(*[('%s-phase Voltage (Mag)' % ph,
                  '%s-phase Voltage (Ang)' % ph) for ph in phases]),
                itertools.chain(*[('%s-sequence Voltage (Mag)' % ph,
                  '%s-sequence Voltage (Ang)' % ph) for ph in ('Zero', 'Positive', 'Negative')]))))
            for n, branch in enumerate(branches):
                sswriter.writerow(list(itertools.chain(branch,
                        (S_3ph.real[n], S_3ph.imag[n]),
                        itertools.chain(*zip(np.real(ph_br_currents[:, n]),
                            np.imag(ph_br_currents[:,n]))),
                        itertools.chain(*zip(np.real(seq_br_currents[:, n]),
                            np.imag(seq_br_currents[:,n]))),
                        itertools.chain(*zip(np.absolute(ph_br_currents[:, n]),
                           np.angle(ph_br_currents[:, n], deg=True))),
                        itertools.chain(*zip(np.absolute(seq_br_currents[:, n]),
                           np.angle(seq_br_currents[:, n], deg=True))))))

            sswriter.writerow(['--------']*28)

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
    seq_voltages = np.array(lineZ.ph_to_seq_v(ph_voltages))
    neg_seq_imbalance = np.abs(seq_voltages[2]/seq_voltages[1])*100

    return ph_voltages, seq_voltages, neg_seq_imbalance


def process_SS_branch_currents(LIS_file, branches, phases=('A', 'B', 'C'),
                            RMS_scale=False):
    ''' Parses LIS_file to get steady state results, then creates vectors of
        branch current phasors on the specified branches. Returns
        ph_br_currents, seq_br_currents as tuple of lists in same order as
        the list of branches.
    '''
    node_voltages, branch_currents = get_SS_results(LIS_file, RMS_scale)

    ph_br_currents = np.array(
        [[branch_currents[(fr_b + p, to_b + p)] for p in phases]
         for fr_b, to_b in branches]).T
    seq_br_currents = np.array(lineZ.ph_to_seq_v(ph_br_currents))
    # Voltages for power calculation
    ph_voltages = np.array(
        [[node_voltages[fr_b + p] for p in phases]
         for fr_b, to_b in branches]).T
    S_3ph = np.sum(ph_voltages * np.conj(ph_br_currents), axis=0) / 1e6

    return ph_br_currents, seq_br_currents, S_3ph


def get_line_params_from_pch(atp_pch_folder, seg_list):
    """ Reads in line parameters from PCH files, saves the data, and combines
        data into an aggregate summary of several parameters. Returns two
        dicts as a tuple:
        seg_data_dict: {seg: params} returns the segment name and
            LineConstPCHCards object for each segment.
        summary_data_dict: Returns various parameters with the line segments
            combined into an equivalent. Parameters returned are the
            following:
            Zsum: Sum of Z matrices of segments
            Ysum: Sum of Y matrices of segments
            Zsum_s, Ysum_s: Symmetrical components of Zsum & Ysum
            ABCD: Transfer matrix in phase quantities
            Zeq: Equivalent Z matrix from ABCD
            Yeq: Equivalent Y matrix from ABCD
            ABCD_s, Zeq_s, Yeq_s: Symmetrical components of prev. three."""
    seg_data_dict = {}
    for seg in seg_list:
        with open(os.path.join(atp_pch_folder, seg.lower() + '.pch')) as \
                pch_file:
            pch_lines = pch_file.readlines()
            params = LineConstPCHCards()
            params.read(pch_lines)
        seg_data_dict[seg] = params

    summary_data_dict = {}
    summary_data_dict['Zsum'] = np.sum([p.Z for _, p in seg_data_dict.items()],
                                       axis=0)
    summary_data_dict['Ysum'] = np.sum([p.Y for _, p in seg_data_dict.items()],
                                       axis=0)
    summary_data_dict['Zsum_s'] = lineZ.ph_to_seq_m(summary_data_dict['Zsum'])
    summary_data_dict['Ysum_s'] = lineZ.ph_to_seq_m(summary_data_dict['Ysum'])
    ABCD_list = [p.ABCD for _, p in seg_data_dict.items()]
    summary_data_dict['ABCD'] = lineZ.combine_ABCD(ABCD_list)
    Z, Y1, Y2 = lineZ.ABCD_to_ZY(summary_data_dict['ABCD'])
    summary_data_dict['Zeq'] = Z
    summary_data_dict['Yeq'] = Y1 + Y2
    summary_data_dict['ABCD_s'] = lineZ.ph_to_seq_m(summary_data_dict['ABCD'])
    Z, Y1, Y2 = lineZ.ABCD_to_ZY(summary_data_dict['ABCD_s'])
    summary_data_dict['Zeq_s'] = Z
    summary_data_dict['Yeq_s'] = Y1 + Y2
    return seg_data_dict, summary_data_dict



def extract_ABCD(ATP_template, ATP_tmp, current_key, switch_key,
                 in_port, out_port,
                 test_current = 500., switch_close_t = '999.',
                 phases =  ('A', 'B', 'C')):

    # ATP_tmp should be in the same directory as ATP_template since most likely
    # the model will include includes of .lib files for line parameters.
    ATP_tmp_full = os.path.join(os.path.dirname(ATP_template), ATP_tmp)

    V1_s = np.zeros((3,3), dtype=np.complex128)
    V1_o = np.zeros((3,3), dtype=np.complex128)
    I1_s = np.zeros((3,3), dtype=np.complex128)
    I1_o = np.zeros((3,3), dtype=np.complex128)
    V2_o = np.zeros((3,3), dtype=np.complex128)
    I2_s = np.zeros((3,3), dtype=np.complex128)

    for out_port_short in (True, False):
        for n, ph in enumerate(phases):
            shutil.copyfile(ATP_template, ATP_tmp_full)
            # Find/replace code numbers in template ATP file and copy to new file
            for n2, ph2 in enumerate(phases):
                replace_text(ATP_tmp_full, current_key,
                        ('%6f.' % (test_current if n2==n else test_current/1000.)),
                        n=n+1)
            replace_text(ATP_tmp_full, switch_key, '-1' if out_port_short else '1')

            # Run ATP on new file
            run_ATP(ATP_tmp_full)

            # Extract steady-state results
            LIS_results = lis_filename(ATP_tmp_full)
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
