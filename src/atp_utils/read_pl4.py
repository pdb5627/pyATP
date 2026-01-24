# -*- coding: utf-8 -*-
#
#  Code originally from
#  lib_readPL4.py
#  https://github.com/ldemattos/readPL4
#  
#  Copyright 2019 Leonardo M. N. de Mattos <l@mattos.eng.br>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 3.0.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

import mmap
import struct
import pandas as pd
import numpy as np

# Read PISA's binary PL4
def read_pl4(pl4file):
    
    miscData = {
        'deltat':0.0,
        'nvar':0,
        'pl4size':0,
        'steps':0,
        'tmax':0.0
    }

    # open binary file for reading
    with open(pl4file, 'rb') as f:
        pl4 = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # read DELTAT
        miscData['deltat'] = struct.unpack('<f', pl4[40:44])[0]
    
        # read number of vars
        miscData['nvar'] = struct.unpack('<L', pl4[48:52])[0] // 2
        
        # read PL4 disk size
        miscData['pl4size'] = struct.unpack('<L', pl4[56:60])[0]-1
        
        # compute the number of simulation miscData['steps'] from the PL4's file size
        miscData['steps'] = (miscData['pl4size'] - 5*16 - miscData['nvar']*16) // \
                            ((miscData['nvar']+1)*4)
        miscData['tmax'] = (miscData['steps']-1)*miscData['deltat']
        
        # generate pandas dataframe to store the PL4's header
        header_data = []
        
        for i in range(0,miscData['nvar']):
            pos = 5*16 + i*16
            h = struct.unpack('3x1c6s6s',pl4[pos:pos+16])
            header_data.append({
            	'TYPE': int(h[0]),
                'FROM': h[1].decode('utf-8'),  # TODO: Is it really utf-8?
                'TO': h[2].decode('utf-8'),
            })
        dfHEAD = pd.DataFrame(header_data)

        # Check for unexpected rows of zeroes
        # See https://github.com/ldemattos/readPL4/issues/2
        expsize = (5 + miscData['nvar'])*16 + miscData['steps']*(miscData['nvar']+1)*4
        nullbytes = 0
        if miscData['pl4size'] > expsize: 
            nullbytes = miscData['pl4size']-expsize
        
        # read and store actual data, map it to a numpy read only array
        data = np.memmap(f,dtype=np.float32,mode='r',shape=(miscData['steps'],miscData['nvar']+1),offset=(5 + miscData['nvar'])*16 + nullbytes)
            
        return dfHEAD,data,miscData

# Convert types from integers to strings
def convertType(df):    
    df['TYPE'] = df['TYPE'].apply(lambda x: 'V-node' if x == 4 else x)
    df['TYPE'] = df['TYPE'].apply(lambda x: 'E-bran' if x == 7 else x)
    df['TYPE'] = df['TYPE'].apply(lambda x: 'V-bran' if x == 8 else x)
    df['TYPE'] = df['TYPE'].apply(lambda x: 'I-bran' if x == 9 else x)
    
    return 0
    
# Get desired variable data
def get_var_data(dfHEAD, data, Type, From, To):
    
    # Search for desired data in header
    df = dfHEAD[(dfHEAD['TYPE'] == Type) & (dfHEAD['FROM'] == From) & (dfHEAD['TO'] == To)]
                
    if not df.empty:
        data_sel = data[:, df.index.values[0]+1] # One column shift given time vector
        
    else:
        print("Variable %s-%s of %s not found!"%(From,To,Type))
        return(None)

    return(data_sel)


"""
Information about the meaning of the type codes in the PL4 file is inferred from 
code in saveToPl4File function in the PlotXY open-source code:
https://github.com/max-privato/PlotXY_OpenSource/blob/master/CSimOut.cpp
"""
type_map = {
    1: 's', # SM Variable
    2: 't', # TACS variable
    3: 'm', # MODELS variable
    4: 'v', # Node voltage
    5: 'u', # UM variable
    6: 'p', # Branch power
    7: 'e', # Branch energy
    8: 'v', # Branch voltage
    9: 'c'} # Branch current


def map_name(col_type, col_from, col_to):
    name_parts = [type_map[col_type]]
    name_parts.append(col_from.replace(' ', ''))
    if col_to.strip():
        name_parts.append(col_to.replace(' ', ''))
    return ':'.join(name_parts)


def pl4_to_dataframe(filename):
    """
    Read PISA-formatted binary PL4 file into a pandas dataframe.

    :param filename: Filename of PL4 to read from.
    :return: Pandas dataframe with data from PL4 file.
    """
    dfHEAD, data, miscData = read_pl4(filename)

    col_names = [map_name(col.TYPE, col.FROM, col.TO) for col in dfHEAD.itertuples()]
    df =pd.DataFrame(data[:, 1:], index=data[:, 0], columns=col_names)
    return df
