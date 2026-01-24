"""
Symmetrical component transformations and ABCD matrix utilities.

This module provides functions for converting between phase and sequence
quantities, and for working with two-port ABCD transmission line matrices.

Functions extracted from the original lineZ module for use by atp_utils.
"""

import cmath

import numpy as np
from numpy.linalg import inv

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
A = np.array([[1., 1., 1.], [1., alpha**2, alpha], [1, alpha, alpha**2]],
         dtype=cdtype)
A_inv = inv(A)

# Define number of phases globally to avoid having to compute it elsewhere,
# especially in functions that loop extensively.
n_ph = 3


def ph_to_seq_v(ph_vec):
    ''' Convert phase quantities to sequence components.
        Call on vectors.'''
    n_ph = ph_vec.shape[0]
    A_inv_block = np.kron(np.eye(n_ph // 3), A_inv)
    # Vector case
    return A_inv_block.dot(ph_vec)


def ph_to_seq_m(ph_vec):
    ''' Convert phase quantities to sequence components.
        Call on matrices. '''
    n_ph = ph_vec.shape[0]
    A_inv_block = np.kron(np.eye(n_ph // 3), A_inv)
    A_block = np.kron(np.eye(n_ph // 3), A)
    return A_inv_block.dot(ph_vec).dot(A_block)


def seq_to_ph_v(seq_vec):
    ''' Convert sequence quantities to phase components'''
    n_ph = seq_vec.shape[-1]
    A_block = np.kron(np.eye(n_ph // 3), A)
    # Vector case
    return A_block.dot(seq_vec)


def seq_to_ph_m(seq_vec):
    ''' Convert sequence quantities to phase components'''
    n_ph = seq_vec.shape[-1]
    A_block = np.kron(np.eye(n_ph // 3), A)
    # Matrix / array case
    A_inv_block = np.kron(np.eye(n_ph // 3), A)
    return A_block.dot(seq_vec).dot(A_inv_block)


def ZY_to_ABCD(Z, Y):
    ''' Create Two-port ABCD matrix from series impedance Z and shunt admittance Y.
        See Bergen & Vittal _Power Systems Analysis_ p.99-100
        Function allows for multiple (e.g. three) phases per port of the system.'''
    Y1 = Y2 = Y/2.
    return ZY_to_ABCD2(Z, Y1, Y2)


def ZY_to_ABCD2(Z, Y1, Y2):
    ''' Create Two-port ABCD matrix from series impedance Z and shunt admittances Y1 & Y2.
        See Bergen & Vittal _Power Systems Analysis_ p.99-100
        Function allows for multiple (e.g. three) phases per port of the system.'''
    n_ph = Z.shape[0]
    ABCD = np.empty((n_ph*2, n_ph*2), dtype=cdtype)
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
    # Cpi = Y1 + Y1*Z*Y2 + Y2
    # print('C - Cpi', C - Cpi)

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


def combine_ABCD(ABCD_list):
    ''' Cascade ABCD matrices by matrix multiplication.'''
    rtn = np.eye(ABCD_list[0].shape[0], dtype=cdtype)
    for ABCD in ABCD_list:
        rtn = rtn.dot(ABCD)
    return rtn
