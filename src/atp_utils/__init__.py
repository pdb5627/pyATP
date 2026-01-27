"""
atp-utils: Module for working with ATP data files and extracting results.

This package provides utilities for:
- Driving ATP simulations from Python
- Extracting steady-state results from LIS files
- Reading PL4 binary output files
- Parsing LIS file sections (simulation extrema, statistic switching)
- Symmetrical component transformations
"""

__version__ = '0.2.0'

from .core import (
    run_ATP,
    atp_basename,
    lis_filename,
    replace_text,
    get_SS_results,
    output_ss_file,
    extract_ABCD,
    process_SS_bus_voltages,
    process_SS_branch_currents,
    get_line_params_from_pch,
    make_ATPline_lib,
    ATP_path,
    ATP_exe,
)

from .line_const_parsing import (
    LineConstCards,
    LineConstPCHCards,
)
from .read_pl4 import read_pl4, pl4_to_dataframe
from . import lis_parsing
from . import sequence_components
