import numpy as np
import pandas as pd
import re
import xarray as xr

from .common import infer_phase, get_output_variables


"""
The list of switches:

is a listing of switches that are to have closing or opening times varied randomly.
          Entry    Switch      From        To         Columns 15-24        Columns 25-34       Reference switch number
         number    number       bus       bus          (in seconds)         (in seconds)       (0 implies independence)
              1        22    X0001A    BRKR1A             0.012500             0.002405               0
              2        23    X0001B    BRKR1B             0.000000             0.002420              22
              3        24    X0001C    BRKR1C             0.016000             0.002420              22
              
              
Before the list of buses / branches:

Well, almost identical!  Output of the reference angle in degrees has been added at the extreme left in otherwise blank space.
"""
def get_statistical_switches(LIS_file):
    start_signal = ("is a listing of switches that are to have closing or opening times "
        "varied randomly.")
    end_signal = ("In addition to variation of the switching time that is caused by "
        "each switch`s own distribution, there is the added random delay")
    # Scan down to the right section
    i = iter(LIS_file)
    l = next(i)
    while l != start_signal:
        try:
            l = next(i)
        except StopIteration:
            raise ValueError("Statistical switches section not found")
    # Skip two header rows
    next(i)
    next(i)
    
    switches = []
    
    l = next(i)
    while l != end_signal:
        # I *think* that there will always be spaces between the columns.
        # A manually  built ATP file might not?
        parts = l.split()
        from_bus, to_bus, phase = infer_phase(parts[2], parts[3])
        switches.append({
            "switch_number": int(parts[1]),
            "from_bus": from_bus,
            "to_bus": to_bus,
            "phase": phase,
            "time_delay_center": float(parts[4]),
            "time_delay_sigma": float(parts[5]),
            "reference_switch_number": int(parts[6]),
        })
        l = next(i)
    return switches


def get_statistical_switching_results(LIS_file):
    """ Extract statistical switching results from an ATP LIS file.
    
    Switching times, maxima, and times of maxima are saved into an xarray Dataset with
    data variables as follows:
    "switching_times": indexed by tuple of bus names (bus1, bus2)
    "maxima": indexed by 
    """
    start_signal = re.compile("             Random switching times for "
        "simulation number (?P<sim_number>[ 0-9]+) :")
    
    simulation_results = []
    truncated_file = False
    
    i = iter(LIS_file)
    l = next(i)
    while True:
        while not (m := start_signal.match(l)):
            try:
                l = next(i)
            except StopIteration:
                if not simulation_results:
                    raise ValueError("Statistical switching simulation results section not found")
                else:
                    truncated_file = True
                    break
        if truncated_file:
            break
        sim_number = int(m.group("sim_number"))
        # Get switch times for the simulation
        # Need to see what ATP does if there are too many switches to fit on one line. Probably they wrap.
        # It looks like it is followed by a blank line.
        #                                   22  9.3116024E-03   23  7.5364674E-03   24  2.1729123E-02
        table_dumping = "==== Table dumping"
        
        switch_info = []
        l = next(i)
        while l.strip() and table_dumping not in l:
            switch_info.extend(l.split())
            l = next(i)
        switch_numbers = [int(v) for v in switch_info[::2]]
        switch_times = [float(v) for v in switch_info[1::2]]
        
        if table_dumping in l:
            l = next(i)
        
        maxima = []
        l = next(i)
        while "Times of maxima" not in l[:17]:
            # TODO: Look for reference angle in the left margin
            maxima.extend(float(s) for s in l[17:].split())
            try:
                l = next(i)
            except StopIteration:
                truncated_file = True
                break
        if truncated_file:
            break
        
        times_of_maxima = []
        while not start_signal.match(l) and l:
            times_of_maxima.extend(float(s) for s in l[17:].split())
            try:
                l = next(i)
            except StopIteration:
                truncated_file = True
                break
        if truncated_file:
            break
        
        simulation_results.append({
            "simulation_number": sim_number,
            "switch_numbers": switch_numbers,
            "switch_times": switch_times,
            "maxima": maxima,
            "times_of_maxima": times_of_maxima,
        })
        
        # List of all simulation results is followed by a blank line
        if not l:
            break
    
    if truncated_file:
        seed = np.nan
    else:
        l = next(i)
        m = re.match(" MAIN20 dumps OVER12 dice seed  SEEDSV = (?P<seed>[ 0-9E+]+)", l)
        # Keep seed as string. Not sure how it would need to be entered for ATP to use it
        seed = m.group("seed")

    # Prep the MultiIndex for variables
    var_list = get_output_variables(LIS_file)
    var_list = list(zip(*var_list))
    var_list[2] = pd.Categorical(var_list[2], categories=["A", "B", "C", ""], ordered=True)
    mi = pd.MultiIndex.from_arrays(var_list, names=["from_node", "to_node", "phase", "var_type"])

    ds = xr.Dataset(
        data_vars={
            "switch_times": (
                ("simulation_number", "switch_number"),
                np.array([s["switch_times"] for s in simulation_results]),
            ),
            "maxima": (
                ("simulation_number", "model_var"),
                np.array([s["maxima"] for s in simulation_results])
            ),
            "times_of_maxima": (
                ("simulation_number", "model_var"),
                np.array([s["times_of_maxima"] for s in simulation_results])
            ),
        },
        coords={
            "simulation_number": np.array([s["simulation_number"] for s in simulation_results]),
            "switch_number": np.array(simulation_results[0]["switch_numbers"]),
        },
    ).assign_coords(xr.Coordinates.from_pandas_multiindex(mi, "model_var")
    )
    return ds
    
