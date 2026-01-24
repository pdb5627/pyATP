import numpy as np
import re
from collections.abc import Sequence


def infer_phase(from_node, to_node):
    phase_ids={"A", "B", "C"}
    
    # Protect special names from phase identification
    protected_names = {"TERRA"}
    protection_flag = "!!!"
    if to_node in protected_names:
        to_node = to_node + protection_flag
    if from_node in protected_names:
        from_node = from_node + protection_flag
    
    phase = ""
    if from_node and from_node[-1] in phase_ids:
        candidate = from_node[-1]
        if not to_node or to_node.endswith(candidate) or to_node[-1] not in phase_ids:
            phase = candidate
            from_node = from_node[:-1]
            to_node = to_node[:-1] if to_node.endswith(candidate) else to_node
            
    # Undo special name flagging
    if to_node.endswith(protection_flag):
        to_node = to_node[:-len(protection_flag)]
    if from_node.endswith(protection_flag):
        from_node = from_node[:-len(protection_flag)]
    return from_node, to_node, phase

"""
To know about the quantities being logged:

Column headings for the  124   EMTP output variables follow.  These are divided among the 5 possible classes as follows ....
  First  87     output variables are electric-network voltage differences (upper voltage minus lower voltage);
  Next   37     output variables are branch currents (flowing from the upper node to the lower node);
Branch power  consumption (power  flow, if a switch) is treated like a branch voltage for this grouping (3  variables);
Branch energy consumption (energy flow, if a switch) is treated like a branch current for this grouping (3  variables).
   Step      Time      BUS2A      BUS2B      BUS2C      ZNOL2A     ZNOL2B     ZNOL2C     ADAMEC     ADAMEB     ADAMEA     BVCKEB
                       ZNOL2A     ZNOL2B     ZNOL2C     TERRA      TERRA      TERRA

                       BVCKEC     BVCKEA     COLMBA     COLMBB     COLMBC     NROCHA     NROCHB     NROCHC     ADAMWC     ADAMWB
"""
def get_output_variables(LIS_file: Sequence[str]):
    n_vars_re = re.compile(r"^Column headings for the (?P<n_vars>[ 0-9]+) EMTP output variables follow.")
    v_vars_re = re.compile(r"^  First (?P<n_v_vars>[ 0-9]+) output variables are electric-network voltage differences")
    i_vars_re = re.compile(r"^  Next (?P<n_i_vars>[ 0-9]+) output variables are branch currents")
    p_vars_re = re.compile(r"^Branch power  consumption \(power  flow, if a switch\) is treated like "
        r"a branch voltage for this grouping \((?P<n_p_vars>[ 0-9]+) variables\)")
    e_vars_re = re.compile(r"^Branch energy consumption \(energy flow, if a switch\) is treated like "
        r"a branch current for this grouping \((?P<n_e_vars>[ 0-9]+) variables\).")
    
    # Scan down to the right section
    i = iter(LIS_file)
    l = next(i)
    while not (m := n_vars_re.match(l)):
        try:
            l = next(i)
        except StopIteration:
            raise ValueError("Output variable section not found")
            
    # Read in number of vars
    n_vars = int(m.group("n_vars"))
    l = next(i)
    m = v_vars_re.match(l)
    n_v_vars = int(m.group("n_v_vars"))
    l = next(i)
    m = i_vars_re.match(l)
    n_i_vars = int(m.group("n_i_vars"))
    l = next(i)
    m = p_vars_re.match(l)
    n_p_vars = int(m.group("n_p_vars"))
    l = next(i)
    m = e_vars_re.match(l)
    n_e_vars = int(m.group("n_e_vars"))
    
    
    vars_list = []
    l1 = ""
    # Each from/to pair of lines is followed by a blank line except the last one.
    while not l1.strip():
        l1 = next(i)
        l2 = next(i)
        
        # Don't count on trailing spaces. If opened in a programming editor, trailing
        # spaces may be automatically removed.
        idx = 0
        while True:
            var_slice = slice(23 + idx*11, 23 + (idx + 1)*11)
            from_node = l1[var_slice].strip()
            to_node = l2[var_slice].strip()
            if not from_node and not to_node:
                break
            from_node, to_node, phase = infer_phase(from_node, to_node)
            vars_list.append((from_node, to_node, phase))
            idx += 1
            
        l1 = next(i)
    # I don't know any way to tell the difference between a v-var and a p-var from the
    # lis file, short of parsing all the input cards for output requests. At least for
    # now, leave it up to the user to figure it out.
    
    # Add var type
    
    rtn = ([(*v, "vp") for v in vars_list[:n_v_vars]] + 
        [(*v, "ie") for v in vars_list[n_v_vars:]])
    return rtn
