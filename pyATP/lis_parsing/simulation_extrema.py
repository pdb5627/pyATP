import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import re
from collections.abc import Sequence
import warnings
import xarray as xr

from .common import get_output_variables

def get_simulation_extrema_stacked(LIS_files: list[Path]):
    if not LIS_files:
        raise ValueError("LIS_files must be non-empty")
        
    ds_list = []
    for p in LIS_files:
        try:
            lines = p.read_text(encoding="iso-8859-1").splitlines()
            ds_list.append(get_simulation_extrema(lines))
        except Exception as e:
            warnings.warn(f"Error reading from {p}: {e}")

    ds = xr.concat(
        ds_list,
        dim=xr.IndexVariable("simulation_idx", range(len(ds_list)))
    )
    return ds
    

def get_simulation_extrema(LIS_file):
    """ Extract extrema from simulation (not STATISTIC runs!) from an ATP LIS file.
    
    Maxima, minima, and times of maxima and minima are saved into an xarray Dataset
    with data variables as follows:
    "maxima": indexed by 
    """
    start_signal = ("Extrema of output variables follow.   Order and column positioning "
        "are the same as for the preceding time-step loop output.")
    
    truncated_file = False
    
    i = iter(LIS_file)
    l = next(i)
    while not l == start_signal:
        try:
            l = next(i)
        except StopIteration:
            raise ValueError("Simulation extrema section not found")
    
    sections, sections_for_xarray = list(zip(*[
        ("Variable maxima", "maxima"),
        ("Times of maxima", "times_of_maxima"),
        ("Variable minima", "minima"),
        ("Times of minima", "times_of_minima"),
    ]))

    data = {s: [] for s in sections}
    l = next(i)
    
    for curr_section, next_section in itertools.pairwise(sections + ("BLANK LINE",)):
        while l and next_section not in l[:17]:
            # TODO: Look for reference angle in the left margin
            data[curr_section].extend(float(s) for s in l[17:].split())
            try:
                l = next(i)
            except StopIteration:
                truncated_file = True
                break
        if truncated_file:
            break
    
    # Prep the MultiIndex for variables
    var_list = get_output_variables(LIS_file)
    var_list = list(zip(*var_list))
    var_list[2] = pd.Categorical(var_list[2], categories=["A", "B", "C", ""], ordered=True)
    mi = pd.MultiIndex.from_arrays(var_list, names=["from_node", "to_node", "phase", "var_type"])

    ds = xr.Dataset(
        data_vars={
            s_xr: (("model_var",), np.array(data[s]))
            for s, s_xr in zip(sections, sections_for_xarray)
        },
        coords=xr.Coordinates.from_pandas_multiindex(mi, "model_var"),
    )
    return ds

