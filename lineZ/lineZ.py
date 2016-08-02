import numpy as np
from numpy.linalg import inv, eig, eigh
#import line_profiler

import cmath, itertools
from collections import OrderedDict

# Defining the data type may allow use of a smaller, faster data type
# if the default precision isn't necessary. Or it may allow going with
# a larger datatype if more precision is needed.
nbits = 64
fdtype = np.dtype('float'+str(nbits))
cdtype = np.dtype('complex'+str(2*nbits))

# 120-degree phase shift operator
alpha = cmath.exp(2./3.*np.pi*1.j)

# Symmetrical component transformation matrix
# Converts from symmetrical components to phase quantities
A = np.matrix([[1., 1., 1.], [1., alpha**2, alpha], [1, alpha, alpha**2]], dtype=cdtype)
A_inv = inv(A)
Apos = A[:,1]

def ph_to_seq(ph_vec):
    ''' Convert phase quantities to sequence components. '''
    return A_inv.dot(ph_vec)
    
def seq_to_ph(seq_vec):
    ''' Convert sequence quantities to phase components'''
    return A.dot(seq_vec)


# Define number of phases globally to avoid having to compute it elsewhere,
# especially in functions that loop extensively.
n_ph = 3

def fix_dtype(qty, dtype):
    if qty.dtype != dtype:
        return qty.astype(dtype)
    else:
        return qty

def equivalent_pi(Z, Y, L, print_calcs=False):
    '''Calculate equivalent pi parameters given impedance matrix, admittance matrix, and length'''
    # Get eigenvalues & eigenvectors
    lambda_k, T_v = eig(Z*Y)
    T_i = inv(T_v.T)
    if print_calcs:
        print('lambda_k = ', lambda_k)
        print('T_v = ', T_v)
        print('T_i = ', T_i)
    
        # Verify diagonalization
        print('Verify diagonalization')
        print('Modal voltage matrix = ', inv(T_v)*Z*Y*T_v)
        print('Modal Y matrix = ', T_v.T*Y*T_v)
        print('Modal Z matrix = ', T_i.T*Z*T_i)
    
    # Convert to de-coupled modal values (EMTP Rulebook Eqns. 4.85b & 4.86a)
    Y_modal = np.diag(T_v.T*Y*T_v)
    Z_modal = np.diag(T_i.T*Z*T_i)
    if print_calcs:
        print('Z_modal = ', Z_modal)
        print('Y_modal = ', Y_modal)
    # Modal propagation constant & characteristic impedance (EMTP Rulebook Eqn. 4.84)
    gamma_modal = np.sqrt(lambda_k)
    Z_char = np.sqrt(Z_modal/Y_modal)
    if print_calcs:
        print('gamma_modal = ', gamma_modal)
        print('Z_char = ', Z_char)
    # Apply hyberbolic correction factor based on propagation constant & length (EMTP Rulebook Eqn. 1.14)
    if print_calcs:
        print('Impedance correction = ', np.abs(np.sinh(gamma_modal*L)/(gamma_modal*L)))
    Z_series = L*Z_modal*np.sinh(gamma_modal*L)/(gamma_modal*L)
    Y_shunt = L*Y_modal*np.tanh(gamma_modal*L/2)/(gamma_modal*L/2)
    # Convert back to phase domain
    Z_phase = T_v*np.diag(Z_series)*T_v.T
    Y_phase = T_i*np.diag(Y_shunt)*T_i.T
    if print_calcs:
        print('Adjusted Z_phase = ', Z_phase)
    
    return fix_dtype(Z_phase, cdtype), fix_dtype(Y_phase, cdtype)

def Pt(phases):
    ''' Creates a phase transposition matrix to rearrange the phases of
        the phase impedance matrix of a three-phase line. The desired phase
        transition is indicated by passing in a list with the numbers 0, 1, 
        and 2 in the desired order.
        
        For example suppose the following transposition is desired for an
        H-frame structure:
           o     o                   o     o
         / |     | \               / |     | \
        ---+-----+---             ---+-----+---
        !  |  !  |  !             !  |  !  |  !
        0  |  1  |  2             2  |  0  |  1
           |     |      ===>>>       |     |
           |     |                   |     |
           |     |                   |     |
           |     |                   |     |
        ______________            _____________
        
        The phase list to pass to this function would then be [2, 0, 1].
    '''
    rtn = np.asmatrix(np.zeros((3, 3)), dtype=np.int16)
    for n in range(3):
        rtn[n, phases[n]] = 1
    return rtn

def apply_phasing(M, Pt):
    ''' Permutes rows and columns of input matrix according to order in Pt.
        Pt should be given as a tuple.
    '''
    Pt_l = apply_phasing._arrays[Pt]
    return M[:,Pt_l][Pt_l,:]
# Function data member to pre-allocate numpy arrays. This is somewhat faster
# Than making new
apply_phasing._arrays = {Pt: np.array(Pt) for Pt in itertools.permutations((0, 1, 2)) }
    
def chain_permutations_old(Plist):
    rtn = np.array([0, 1, 2])[Plist[0]]
    for P in Plist[1:]:
        rtn = rtn[P]
    return rtn

def chain_permutations_2nd(Plist):
    rtn = Plist[0]
    for P in Plist[1:]:
        rtn = [rtn[n] for n in P]
    return rtn
    
def chain_permutations_3rd(Plist):
    rtn = Plist[0]
    for P in Plist[1:]:
        rtn = (rtn[P[0]], rtn[P[1]], rtn[P[2]])
    return rtn

def chain_permutations_4th(Plist):
    rtn = Plist[0]
    for P in Plist[1:]:
        rtn = chain_permutations_4th._chain_lookup[rtn][P]
    return rtn
chain_permutations_4th._chain_lookup = {P1: {P2: chain_permutations_3rd((P1, P2))
            for P2 in itertools.permutations((0, 1, 2))}
            for P1 in itertools.permutations((0, 1, 2))}

chain_permutations = chain_permutations_4th

def ZY_to_ABCD(Z, Y):
    ''' Create Two-port ABCD matrix from series impedance Z and shunt admittance Y.
        See Bergen & Vittal _Power Systems Analysis_ p.99-100
        Function allows for multiple (e.g. three) phases per port of the system.'''
    #n_ph = Z.shape[0]
    ABCD = np.asmatrix(np.empty((n_ph*2, n_ph*2), dtype=cdtype))
    ABCD[:n_ph,:n_ph] = np.eye(n_ph, dtype=cdtype) + Z*Y/2
    ABCD[:n_ph,n_ph:] = Z
    ABCD[n_ph:,:n_ph] = Y + Y*Z*Y/4
    ABCD[n_ph:,n_ph:] = np.eye(n_ph, dtype=cdtype) + Y*Z/2

    return ABCD

def ZY_to_ABCD2(Z, Y1, Y2):
    ''' Create Two-port ABCD matrix from series impedance Z and shunt admittances Y1 & Y2.
        See Bergen & Vittal _Power Systems Analysis_ p.99-100
        Function allows for multiple (e.g. three) phases per port of the system.'''
    #n_ph = Z.shape[0]
    ABCD = np.asmatrix(np.empty((n_ph*2, n_ph*2), dtype=cdtype))
    ABCD[:n_ph,:n_ph] = np.eye(n_ph, dtype=cdtype) + Z.dot(Y2)
    ABCD[:n_ph,n_ph:] = Z
    ABCD[n_ph:,:n_ph] = Y1 + Y1.dot(Z).dot(Y2) + Y2
    ABCD[n_ph:,n_ph:] = np.eye(n_ph, dtype=cdtype) + Y1.dot(Z)

    return ABCD
    
def ABCD_to_ZY(ABCD):
    ''' Create pi model from two-port ABCD matrix.
        See Bergen & Vittal _Power Systems Analysis_ p.99-100
        Function allows for multiple (e.g. three) phases per port of the system.
        Returns Z, Y1 (from end), Y2 (to end)'''
    A, B, C, D = ABCD_breakout(ABCD)
    
    Z = B
    Y2 = inv(Z)*(A - np.eye(n_ph))
    Y1 = (D - np.eye(n_ph))*inv(Z)
    # Verify if C is consistent with a pi model
    Cpi = Y1 + Y1*Z*Y2 + Y2
    print('C - Cpi', C - Cpi)

    return Z, Y1, Y2

def ABCD_breakout(ABCD):
    ''' Breaks ABCD matrix out into A, B, C, and D
    '''
    #n_ph = ABCD.shape[0]//2
    A = ABCD[:n_ph, :n_ph]
    B = ABCD[:n_ph, n_ph:]
    C = ABCD[n_ph:, :n_ph]
    D = ABCD[n_ph:, n_ph:]
    return A, B, C, D
    

def impedance_calcs(Zstr, Ystr, L, str_types, Pt_list, print_calcs = False, hyperbolic = True, 
                    shunt = True, Z_w_shunt = True):
    ''' Calculates total line impedance for a series of line segments of potentially different
        construction type and phasing. Input parameters are the following:
        Zstr: Dict of per-unit-length series phase impedance matrices for each structure type
        Ystr: Dict of per-unit-length shunt phase admittance matrices for each structure type
            (only used if hyperbolic = True or shunt = True, otherwise ignored)
        L: List of segment lengths
        str_types: List of structure types used for each segment
        Pt_list: List of phase transitions between segments. First element is phasing of first segment
        print_calcs: When True, prints phase impedances of each segment and total phase impedance
        hyperbolic: When True, applies hyperbolic correction factors to series impedances and shunt admittances
        shunt: When True, an equivalent pi model of the cascaded series of pi representations of the segments
            is calculated.
        Z_w_shunt: When Z_w_shunt is True, the returned total impedance includes
            the shunt admittance at the sending end in parallel with the series impedance. This allows
            comparison with calculations done for example in ATP where current is injected into the line
            at the sending end and the impedance calculated from the voltages at the sending end. If
            Z_w_shunt is False, then the returned impedance is only the series impedance of the equivalent
            pi model for the whole line. This parameter has no effect if shunt = False.
            
        TODO: A recursive implementation of this function could potentially allow memoization to speed
            it up a lot.
    '''
    cum_Pt = [Pt(chain_permutations(Pt_list[:n+1])) for n in range(len(Pt_list))]
    Z_transp = [cum_Pt[n].T*Zstr[str_types[n].rstrip('_*')]*cum_Pt[n] for n in range(len(str_types))]
    Y_transp = [cum_Pt[n].T*Ystr[str_types[n].rstrip('_*')]*cum_Pt[n] for n in range(len(str_types))]
    Z_list, Y_list = zip(*[equivalent_pi(Z, Y, lseg) if hyperbolic else (Z*lseg, Y*lseg)
                           for Z, Y, lseg in zip(Z_transp, Y_transp, L)])
    if shunt:
        N_list = [ZY_to_ABCD(Zseries, Ys) for Zseries, Ys in zip(Z_list, Y_list)]
        #n_ph = N_list[0].shape[0]/2
        Neq = np.eye(n_ph*2, dtype=cdtype)
        for n, N in enumerate(N_list):
            Neq = Neq*N # Cascade for ABCD representation is just matrix multiplication
        
        A, B, C, D = ABCD_breakout(Neq)
        if Z_w_shunt:
            
            Ztotal = B.dot(inv(D))
        else:
            Ztotal = B
        
    else:
        Ztotal = sum(Z_list)
    
    if print_calcs:
        print('Calculated Phase Impedances')
        for n, Zt in enumerate(Z_list):
            print(n, np.absolute(Zt*Apos).T)
        print('Zphase total:', np.absolute(Ztotal*Apos).T)
        #print('Sum of sections Zphase:', np.sum((np.absolute(Zt*Apos) for Zt in Z_list), axis=1).T)
    return Ztotal
    
def impedance_calcs_precompute(Zstr, Ystr, L, str_types, hyperbolic = True, 
                    shunt = True):
    ''' Pre-computes phase impedance matrices for each line section for all six
        possible phasing options. Returns a list of the same length as L where 
        each element of the list is a dict of either ABCD matrices (if shunt is
        True) or phase impedance matrices (if shunt is False).
        
        Input parameters are the following:
        Zstr: Dict of per-unit-length series phase impedance matrices for each structure type
        Ystr: Dict of per-unit-length shunt phase admittance matrices for each structure type
            (only used if hyperbolic = True or shunt = True, otherwise ignored)
        L: List of segment lengths
        str_types: List of structure types used for each segment
        hyperbolic: When True, applies hyperbolic correction factors to series impedances and shunt admittances
        shunt: When True, ABCD matrices of an equivalent pi model of the segments
            is calculated and returned.
            
        This implementation attempts to speed up the computations by taking the
        hyperbolic correction and phase permutations out of the loop. Instead they
        are pre-computed and stored in a look-up table.
    '''
    all_phasing_opts = ((0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,1,0), (2,0,1))
    #n_ph = Zstr[str_types[0].rstrip('_*')].shape[0]
    if shunt:
        Pt_mat_dict = {P: np.bmat([[Pt(P), np.zeros((n_ph, n_ph))], [np.zeros((n_ph, n_ph)), Pt(P)]]).astype('int16')
                            for P in all_phasing_opts}
    else:
        Pt_mat_dict = {P: Pt(P)
                        for P in all_phasing_opts}
    
    rtn = [{} for _ in L]
    '''z_list = [Zstr[str.rstrip('_*')] for str in str_types]
    y_list = [Ystr[str.rstrip('_*')] for str in str_types]
    ZY_list = [equivalent_pi(Z, Y, lseg) if hyperbolic else (Z*lseg, Y*lseg)
                           for Z, Y, lseg in zip(z_list, y_list, L)]'''
    # Clear cache if it is the wrong length or doesn't exist
    try:
        if len(impedance_calcs_precompute.cache) != len(rtn):
            raise AttributeError
    except AttributeError:
        impedance_calcs_precompute.cache = [{} for _ in L]
        
    for n, str in enumerate(str_types):
        try:
            lseg = L[n]
            rtn[n] = impedance_calcs_precompute.cache[n][lseg]
        except (KeyError, AttributeError):
            Z = Zstr[str.rstrip('_*')]
            Y = Ystr[str.rstrip('_*')]
            ZY = equivalent_pi(Z, Y, lseg) if hyperbolic else (Z*lseg, Y*lseg)
        
            if shunt:
                rtn[n] = { P: Pt_mat.T*ZY_to_ABCD(*ZY)*Pt_mat for P, Pt_mat in Pt_mat_dict.items() }
                
            else:
                
                rtn[n] = { P: Pt_mat.T*Z*Pt_mat for P, Pt_mat in Pt_mat_dict.items() }
            impedance_calcs_precompute.cache[n][lseg] = rtn[n]

    return rtn

def impedance_calcs_from_precompute(precomputed_list, Pt_list, print_calcs = False,  
                    shunt = True, Z_w_shunt = True):
    ''' Calculates total line impedance for a series of line segments of potentially different
        construction type and phasing. Input parameters are the following:
        precomputed_list: List of dicts of either ABCD matrices (if shunt is True)
            or phase impedance matrices (if shunt is False).

        Pt_list: List of phase transitions between segments. First element is phasing of first segment
        print_calcs: When True, prints phase impedances of each segment and total phase impedance
        shunt: When True, an equivalent pi model of the cascaded series of pi representations of the segments
            is calculated.
        Z_w_shunt: When Z_w_shunt is True, the returned total impedance includes
            the shunt admittance at the sending end in parallel with the series impedance. This allows
            comparison with calculations done for example in ATP where current is injected into the line
            at the sending end and the impedance calculated from the voltages at the sending end. If
            Z_w_shunt is False, then the returned impedance is only the series impedance of the equivalent
            pi model for the whole line. This parameter has no effect if shunt = False.
       
       This is the implementation that uses precomputed impedance or ABCD matrices
       for each segment to try to speed up the loop over combinations of phasing.

    '''
    cum_Pt = [tuple(chain_permutations(Pt_list[:n+1])) for n in range(len(Pt_list))]

    if shunt:
        N_list = [precomputed_list[n][P] for n, P in enumerate(cum_Pt)]
        #n_ph = N_list[0].shape[0]//2

        Neq = np.eye(n_ph*2, dtype=N_list[0].dtype)
        for n, N in enumerate(N_list):
            Neq = Neq*N # Cascade for ABCD representation is just matrix multiplication

        A, B, C, D = ABCD_breakout(Neq)
        if Z_w_shunt:
            
            Ztotal = B.dot(inv(D))
        else:
            Ztotal = B
        
    else:
        Ztotal = sum(precomputed_list[n][P] for n, P in enumerate(cum_Pt))
  
    return Ztotal

def impedance_calcs_from_precompute_recursive(precomputed_list, Pt_list, print_calcs = False,  
                    shunt = True, Z_w_shunt = True, rtn_ABCD = False):
    ''' Calculates total line impedance for a series of line segments of potentially different
        construction type and phasing. Input parameters are the following:
        precomputed_list: List of dicts of either ABCD matrices (if shunt is True)
            or phase impedance matrices (if shunt is False).

        Pt_list: List of phase transitions between segments. First element is phasing of first segment
        print_calcs: When True, prints phase impedances of each segment and total phase impedance
        shunt: When True, an equivalent pi model of the cascaded series of pi representations of the segments
            is calculated.
        Z_w_shunt: When Z_w_shunt is True, the returned total impedance includes
            the shunt admittance at the sending end in parallel with the series impedance. This allows
            comparison with calculations done for example in ATP where current is injected into the line
            at the sending end and the impedance calculated from the voltages at the sending end. If
            Z_w_shunt is False, then the returned impedance is only the series impedance of the equivalent
            pi model for the whole line. This parameter has no effect if shunt = False.
       
       This is the implementation that uses precomputed impedance or ABCD matrices
       for each segment to try to speed up the loop over combinations of phasing.

    '''

    if shunt:
        Neq = precomputed_list[0][Pt_list[0]]
        if len(Pt_list) > 1:
            tmp = chain_permutations(Pt_list[:2])
            Neq = np.dot(Neq, impedance_calcs_from_precompute_recursive(precomputed_list[1:],
                        (tmp,) + Pt_list[2:],
                        print_calcs = print_calcs,  
                        shunt = shunt,
                        Z_w_shunt = Z_w_shunt,
                        rtn_ABCD = True))
        if rtn_ABCD:
            return Neq                

        #n_ph = Neq.shape[0]//2

        A, B, C, D = ABCD_breakout(Neq)
        if Z_w_shunt:
            
            Ztotal = B.dot(inv(D))
        else:
            Ztotal = B
        
    else:
        Ztotal = precomputed_list[0][Pt_list[0]]
        if len(Pt_list) > 1:
            Ztotal = Ztotal + impedance_calcs_from_precompute_recursive(precomputed_list[1:],
                        chain_permutations(Pt_list[:2]) + Pt_list[2:],
                        print_calcs = print_calcs,  
                        shunt = shunt,
                        Z_w_shunt = Z_w_shunt,
                        rtn_ABCD = False)
  
    return Ztotal
    
#@Memoized
def impedance_calcs_recursive(Zstr, Ystr, L, str_types, Pt_list, print_calcs = False, hyperbolic = True, 
                    shunt = True, Z_w_shunt = True, rtn_ABCD = False):
    ''' Calculates total line impedance for a series of line segments of potentially different
        construction type and phasing. Input parameters are the following:
        Zstr: Dict of per-unit-length series phase impedance matrices for each structure type
        Ystr: Dict of per-unit-length shunt phase admittance matrices for each structure type
            (only used if hyperbolic = True or shunt = True, otherwise ignored)
        L: List of segment lengths
        str_types: List of structure types used for each segment
        Pt_list: List of phase transitions between segments. First element is phasing of first segment
        print_calcs: When True, prints phase impedances of each segment and total phase impedance
        hyperbolic: When True, applies hyperbolic correction factors to series impedances and shunt admittances
        shunt: When True, an equivalent pi model of the cascaded series of pi representations of the segments
            is calculated.
        Z_w_shunt: When Z_w_shunt is True, the returned total impedance includes
            the shunt admittance at the sending end in parallel with the series impedance. This allows
            comparison with calculations done for example in ATP where current is injected into the line
            at the sending end and the impedance calculated from the voltages at the sending end. If
            Z_w_shunt is False, then the returned impedance is only the series impedance of the equivalent
            pi model for the whole line. This parameter has no effect if shunt = False.
            
        This is an attempt at a recursive implementation of this function in hopes of gaining speed
        from memoization.
        
        TODO: Could the hyperbolic & equivalent pi calcs be done without including
              phasing? This would allow them to be done ONCE and then only apply
              phase permutation matrices. Phase permutations of pi model could
              likewise be pre-computed for each section with a lookup in loop
              of phasing options.
    '''
    Pt_mat = Pt(Pt_list[0])
    Z_transp = Pt_mat.T*Zstr[str_types[0].rstrip('_*')]*Pt_mat
    if hyperbolic or shunt:
        Y_transp = Pt_mat.T*Ystr[str_types[0].rstrip('_*')]*Pt_mat
    Zseries, Ys = equivalent_pi(Z_transp, Y_transp, L[0]) if hyperbolic \
                    else (Z_transp*L[0], Y_transp*L[0])
                          
    if shunt:
        Neq = ZY_to_ABCD(Zseries, Ys)
        if L[1:].size > 0:
            Neq *= impedance_calcs_recursive(Zstr, Ystr, L[1:], str_types[1:],
                                             [chain_permutations(Pt_list[:2])] + Pt_list[2:],
                                             print_calcs = print_calcs, hyperbolic = hyperbolic, 
                                             shunt = shunt, Z_w_shunt = Z_w_shunt, rtn_ABCD = True)
        if rtn_ABCD:
            return Neq
        #n_ph = Neq.shape[0]/2
        A, B, C, D = ABCD_breakout(Neq)
        if Z_w_shunt:
            
            Ztotal = B.dot(inv(D))
        else:
            Ztotal = B
        
    else:
        Ztotal = Z_transp
        if L[1:].size > 0:
            Ztotal += impedance_calcs_recursive(Zstr, Ystr, L[1:], str_types[1:],
                                             [chain_permutations(Pt_list[:2])]+Pt_list[2:],
                                             print_calcs = print_calcs, hyperbolic = hyperbolic, 
                                             shunt = shunt, Z_w_shunt = Z_w_shunt, rtn_ABCD = False)
    
    if print_calcs:
        print('Zphase total:', np.absolute(Ztotal*Apos).T)
    return Ztotal

def impedance_imbalance(Z):
    ''' Impedance imbalance calculated as the maximum phase impedance deviation from the mean
        phase impedance, divided by the mean phase impedance. Returns factor in %.'''
    Zph = np.absolute(Z*Apos)
    mn = np.mean(Zph)
    return np.max(np.absolute(Zph - mn))/mn*100.

def neg_seq_voltage(Z, Iload=600., Vbase=345.E3):
    Zs = A_inv*Z*A
    return abs(Zs[2,1])*Iload*1.732/Vbase*100.

def neg_seq_unbalance_factor(Z):
    ''' Negative-sequence unbalance factor calculated based on EPRI Redbook Eqn. 3.4.35.
        Returns factor in %.'''
    Zs = A_inv*Z*A
    return abs(Zs[2,1]/Zs[1,1])*100.

def zero_seq_unbalance_factor(Z):
    ''' Zero-sequence unbalance factor calculated based on EPRI Redbook Eqn. 3.4.34.
        Returns factor in %.'''
    Zs = A_inv*Z*A
    return abs(Zs[0,1]/Zs[1,1])*100.

def filter_nondominated_results_old(results, criteria=[impedance_imbalance, neg_seq_unbalance_factor], beat_factor=1.0):
    ''' Return list of results that are non-dominated according to a specified list of criteria.
        The criteria should be functions that take the phase impedance matrix as input and evaluate
        such that LESS is BETTER. The default criteria are phase impedance imbalance and negative-
        sequence unbalance factor criteria.
        Takes a beat_factor argument that allows a result to be dominated unless it beats all other
        non-dominated solutions by at least beat_factor times the criteria results for all criteria.
        The default beat_factor is 1, which results in direct comparison.
        Results list is assumed to be passed in as a list of tuples where the third element is the
        phase impedance matrix.'''
    non_dominated = []
    dominated = set()
    c_r = [[c(r[2]) for c in criteria] for r in results] # Precompute function values
    for n1, c_r1 in enumerate(c_r):
        # Check to see if r1 is dominated by any
        for n2, c_r2 in filter(lambda r: r[0]!=n1 and r[0] not in dominated, enumerate(c_r)):
            if all(beat_factor*cx_r2 < cx_r1 for cx_r1, cx_r2 in zip(c_r1, c_r2)):
                # r1 was dominated by r2
                dominated.add(n1)
                break
        else:
            # r1 was not dominated
            non_dominated.append(n1)
    return [results[n] for n in non_dominated]
    
def filter_nondominated_results_old(results, criteria=[impedance_imbalance, neg_seq_unbalance_factor], precompute=None, beat_factor=1.0):
    ''' Return list of results that are non-dominated according to a specified list of criteria.
        The criteria should be functions that take the phase impedance matrix as input and evaluate
        such that LESS is BETTER. The default criteria are phase impedance imbalance and negative-
        sequence unbalance factor criteria.
        Precomputed criteria values can be passed in using the precompute parameter.
        If given, precompute should be set to an iterable of the same length as results
        with each item containing an iterable of criteria values.
        Takes a beat_factor argument that allows a result to be dominated unless it beats all other
        non-dominated solutions by at least beat_factor times the criteria results for all criteria.
        The default beat_factor is 1, which results in direct comparison.
        Results list is assumed to be passed in as a list of tuples where the third element is the
        phase impedance matrix.'''
    non_dominated = []
    dominated = set()
    if precompute is not None:
        c_r = precompute
    else:
        c_r = [[c(r[2]) for c in criteria] for r in results] # Compute function values
    for n1, c_r1 in enumerate(c_r):
        # Check to see if r1 is dominated by any
        for n2, c_r2 in enumerate(c_r):
            if n2 not in dominated and n2 != n1 and \
                    all(beat_factor*cx_r2 <= cx_r1 for cx_r1, cx_r2 in zip(c_r1, c_r2)):
                # r1 was dominated by r2
                dominated.add(n1)
                break
        else:
            # r1 was not dominated
            non_dominated.append(n1)
    return [results[n] for n in non_dominated]
    
def filter_nondominated_results(results, criteria=[impedance_imbalance, neg_seq_unbalance_factor], precompute=None, beat_factor=1.0):
    ''' Return list of results that are non-dominated according to a specified list of criteria.
        The criteria should be functions that take the phase impedance matrix as input and evaluate
        such that LESS is BETTER. The default criteria are phase impedance imbalance and negative-
        sequence unbalance factor criteria.
        Precomputed criteria values can be passed in using the precompute parameter.
        If given, precompute should be set to an iterable of the same length as results
        with each item containing an iterable of criteria values.
        Takes a beat_factor argument that allows a result to be dominated unless it beats all other
        non-dominated solutions by at least beat_factor times the criteria results for all criteria.
        The default beat_factor is 1, which results in direct comparison.
        Results list is assumed to be passed in as a list of tuples where the third element is the
        phase impedance matrix.'''
    unchecked = set(range(len(results)))
    non_dominated = set()
    dominated = set()
    if precompute is not None:
        c_r = precompute
    else:
        c_r = [[c(r[2]) for c in criteria] for r in results] # Compute function values
    
    while unchecked:
      
        n1 = unchecked.pop()
        # Check best solutions first
        for n2 in non_dominated:
            if all(beat_factor*cx_r2 <= cx_r1 for cx_r1, cx_r2 in zip(c_r[n1], c_r[n2])):
                # n1 was dominated by n2
                dominated.add(n1)
                break
        else:
            # Not dominated by a non-dominated solution yet, check untested solutions
            for n2 in unchecked:
                if all(beat_factor*cx_r2 <= cx_r1 for cx_r1, cx_r2 in zip(c_r[n1], c_r[n2])):
                    # n1 was dominated by n2
                    dominated.add(n1)
                    break
            else:
                # n1 not dominated by any other solution
                non_dominated.add(n1)
    
    return [results[n] for n in non_dominated]

def new_results_dict(soln_list, model_list):
    ''' Initialize to a new results dict suitable for the multimodel criteria
        evaluation of this module. Dict is implemented as an ordered dict
        so that the order of iteration will be guaranteed.
    '''
    results_dict = OrderedDict()
    for soln in soln_list:
        results_dict[soln] = OrderedDict()
        for model in model_list:
            results_dict[soln][model] = None 
    return results_dict

def filter_nondominated_results_multimodel(results, criteria, precompute=False, beat_factor=1.0):
    ''' A different interface to the nondominated_results function that allows
        results to be collected in a data structure with multiple models run
        with the same possible solutions and same criteria to be applied to
        the model results. If the model has multiple levels of variety, then
        these can be combined using a tuple, named tuple, string concatenation,
        etc. Solutions (soln) can be any kind of immutable suitable for indexing.
        
        For each model run, the results should be saved as
        results[soln][model] = (model_results, 
                                criteria_precompute,
                                other_info_precompute)
                                
        results can be initialized by the following if the solution list is
        already available, which it generally should be:
        results = {s: {} for s in soln_list}
    '''
    soln_list, c_r = apply_criteria_multimodel(results, criteria)
    filtered_soln_list = filter_nondominated_results(soln_list, None,
                                                     precompute=c_r,
                                                     beat_factor=beat_factor)
    filtered_results_dict = {}
    for soln in filtered_soln_list:
        filtered_results_dict[soln] = results[soln]
    return filtered_results_dict

def apply_criteria_multimodel(results, criteria, precompute=False):
    ''' Apply a list of criteria functions to results in two-level dict format.
        Returns list of solutions and list of evaluated criteria values.
    '''
    soln_list = list(results.keys())
    # Save list of models to ensure we iterate over them in a consistent order.
    model_list = list(results[soln_list[0]].keys())
        
    c_r = []
    for soln in soln_list:
        c_r.append([])
        for model in model_list:
            if precompute:
                c_r[-1].extend(results[soln][model][1])
            else:
                c_r[-1].extend([c(results[soln][model][0]) for c in criteria])
    return soln_list, c_r
    
def apply_criteria_weighting(results, criteria, model_weights, criteria_weights, precompute=False):
    ''' Apply a list of criteria functions to results in two-level dict format
        and then apply a vector of weights to criteria values computed.
        Returned value will be the solution list and a numpy array of weighted sums.
    '''
    soln_list, c_r = apply_criteria_multimodel(results, criteria, precompute=False)
    weight_vec = [c_wt*m_wt for m_wt, c_wt in itertools.product(model_weights, criteria_weights)]
    return soln_list, np.array(c_r).dot(weight_vec)
    
def weighted_optimum(results, criteria, model_weights, criteria_weights):
    ''' Applies a list of criteria functions to results in two-level dict
        format. Criteria are combined using a weighting vector, and the
        optimum solution is returned.
    '''
    soln_list, c_r_weighted = apply_criteria_weighting(results, criteria, model_weights, criteria_weights)
    idx_optimum = np.argmin(c_r_weighted)
    return soln_list[idx_optimum]

def count_transpositions(phasing_list, str_types):
    ''' Initial implementation of fuction to count transpositions in a phasing list based
        on the structure types.
        Execution time to filter all transition combinations for relatively small test case: 8.09 s
    '''
    cnt = 0
    for ph, ahead, behind in zip(phasing_list[1:], str_types[:-1], str_types[1:]):
        if tuple(ph) != (0, 1, 2) and behind.rstrip('_*') == ahead.rstrip('_*'):
            cnt += 1
    return cnt

def make_tr_list(str_types):
    ''' Make list of possible transposition points based on structure types. Transposition points are places
        where the conductor positions change but NOT transitions from one structure type to another.
    '''
    return list(filter(lambda n: str_types[n-1].rstrip('_*') == str_types[n].rstrip('_*'), range(1, len(str_types))))

def count_transpositions2(phasing_list, tr_list, count_max=9999):
    ''' Improved implementation of function to count transpositions in a phasing list based
        on the structure types. Speed is improved by only by only testing possible transposition
        points, which are pre-computed using make_tr_list, and stopping checks once count_max is reached.
        Execution time to filter all transition combinations for relatively small test case: 1.98 s
    '''
    ctr = 0
    for n in tr_list:
        if phasing_list[n] != (0, 1, 2):
            ctr += 1
            if ctr == count_max:
                return ctr
    return ctr

def make_transitions_dict(transitions_list, str_types, max_transp=None):
    ''' Build transitions_dict by generating phasing combinations with number of
        transpositions up to max_transp. This saves time over generating all
        possible phasing combinations and then filtering most of them out after
        counting transpositions.
        Execution time to build dict for relatively small test case: 0.12 s
    '''
    # List of possible transposition points based on structure type
    transp_list = list(filter(lambda n: str_types[n-1].rstrip('_*') == str_types[n].rstrip('_*'), range(1, len(str_types))))
    if max_transp is None:
        max_transp = len(transp_list)
    
    # List of structure transitions points based on structure type
    tran_list = [0] + list(filter(lambda n: str_types[n-1].rstrip('_*') != str_types[n].rstrip('_*'), range(1, len(str_types))))
    
    transitions_dict = { n: [] for n in range(max_transp+1)}
    
    # 
    transitions_list2 = [list(t) for t in transitions_list]
    for n in transp_list:
        transitions_list2[n].remove((0, 1, 2))

    
    for n in range(max_transp+1):
        for transp in itertools.combinations(transp_list, n):
            transitions_list3 = [(tl[0],) for tl in transitions_list]
            for i in itertools.chain(transp, tran_list):
                transitions_list3[i] = transitions_list2[i]
            transitions_dict[n].extend(itertools.product(*transitions_list3))
    
    return transitions_dict    
    
def print_results(results, sections, str_types, Pos, Str_names, Iload=600., Vbase=345.):
    if len(L)>0:
        PIs = [sum(L[:n]) for n in range(len(L)+1)]
    print('-'*80)
    for r in results:
        print('Number of transpositions: %d' % count_transpositions(r[1], str_types))
        print('Phase impedance imbalance: %.4f %%' % impedance_imbalance(r[2]))
        print('Negative-sequence unbalance factor: %.4f %%' % neg_seq_unbalance_factor(r[2]))
        print('Zero-sequence unbalance factor: %.4f %%' % zero_seq_unbalance_factor(r[2]))
        print('Negative-sequence voltage @ %.0f A: %.4f %%' % (Iload, neg_seq_voltage(r[2], Iload, 345E3)))
    
        print('Transpositions: ', r[1])
        if len(r)>3:
            print('Additional information: ', r[3:])
        
        phasing_info = Pt_list_to_phasing(r[1], str_types, Pos)
        print('Line Sections:')
        for n, phasing, s_start, s_end in zip(range(1,len(sections)), phasing_info, sections[:-1], sections[1:]):
            print(('Mile %.3f: %s' % (s_start[0], s_start[2])) + (' (Transposition)' if n > 1 and sections[n-2][1].rstrip('_*') == sections[n-1][1].rstrip('_*') and r[1][n-1]!=(0, 1, 2) else '')) # + (str(sections[n-2]) if n>1 else '') + str(sections[n-1]) + str(r[1][n-1]))
            print('Section %d: %s (%.3f mi)' % (n, Str_names[s_start[1]], s_end[0] - s_start[0]))
            print(phasing)
        print('Mile %.3f: %s' % (s_end[0], s_end[2]))
        #impedance_calcs(Zstr,  Ystr, L, str_types, r[1], print_calcs = True)
        print('-'*80)

def cum_Pt(Pt_list):
    return [chain_permutations(Pt_list[:n+1]) for n in range(len(Pt_list))]
    
def Pt_list_to_phasing(Pt_list, str_types, Pos, Phase_list=None):
    cum_Pt = [chain_permutations(Pt_list[:n+1]) for n in range(len(Pt_list))]
    if Phase_list is None:
        Phase_list = ('PH1', 'PH2', 'PH3')
    return ['\n'.join([' '*4 + ph + ' - ' + pos
                for ph, pos in zip((Phase_list[n] for n in ph_list),
                    pos_list)])
            for ph_list, pos_list in zip(cum_Pt, [Pos[str_type.rstrip('_*')] for str_type in str_types])]    