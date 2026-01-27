import numpy as np

# Defining the data type may allow use of a smaller, faster data type
# if the default precision isn't necessary. Or it may allow going with
# a larger datatype if more precision is needed.
nbits = 64
fdtype = np.dtype('float'+str(nbits))
cdtype = np.dtype('complex'+str(2*nbits))
