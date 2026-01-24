#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sequence_components
----------------------------------

Tests for `atp_utils.sequence_components` module.
"""

from __future__ import print_function, unicode_literals

import pytest


from atp_utils import sequence_components
import numpy as np


def assert_round_equals(n1, n2, **kwargs):
    if isinstance(n1, dict):
        # Check all keys exist
        for k in n1:
            assert k in n2
        # Check key existence the other way and verify values
        for k in n2:
            assert_round_equals(n1[k], n2[k], **kwargs)
        return
    try:
        np.allclose(n1, n2, equal_nan=True)
    except (TypeError):
        # Types that can't be rounded are checked for equivalence
        assert n1 == n2


balanced_three_phase = np.array([1., sequence_components.alpha**2, sequence_components.alpha])
balanced_three_phase_seq = np.array([0., 1., 0])
balanced_double_three_phase = np.tile(balanced_three_phase, 2)
balanced_double_three_phase_seq = np.tile(balanced_three_phase_seq, 2)

# Examples from Bergen & Vittal _Power Systems Analysis_, p 474
Z_abc_BergenVittal = np.array([[17.304 + 83.562j, 5.717 + 37.81j,
                                5.717 + 32.76j],
                               [5.717 + 37.81j, 17.304 + 83.562j,
                                5.717 + 37.81j],
                               [5.717 + 32.76, 5.717 + 37.81j,
                                17.304 + 83.562]])

Z_s_BergenVittal = np.array([[28.74 + 155.82j, 1.46 - 0.84j, -1.46 - 0.84j],
                             [-1.46 - 0.84j, 11.59 - 47.44j, -2.92 + 1.68j],
                             [1.46 - 0.84j, 2.92 + 1.68j, 11.59 + 47.44j]])

@pytest.mark.parametrize("qty_ph, qty_seq",
                         [(balanced_three_phase, balanced_three_phase_seq),
                          (balanced_double_three_phase,
                           balanced_double_three_phase_seq),
                          (Z_abc_BergenVittal, Z_s_BergenVittal)])
class TestPhToSeq:
    def test_ph_to_seq(self, qty_ph, qty_seq):
        assert_round_equals(sequence_components.ph_to_seq_m(qty_ph), qty_seq)

    def test_seq_to_ph(self, qty_seq, qty_ph):
        assert_round_equals(sequence_components.ph_to_seq_m(qty_ph), qty_seq)

    def test_round_trip_ph(self, qty_seq, qty_ph):
        assert_round_equals(sequence_components.seq_to_ph_m(sequence_components.ph_to_seq_m(qty_ph)), qty_ph)

    def test_round_trip_seq(self, qty_seq, qty_ph):
        assert_round_equals(sequence_components.ph_to_seq_m(sequence_components.seq_to_ph_m(qty_seq)), qty_seq)
