import atp_utils.lis_parsing.statistic_switching as ss
import numpy as np
import pandas as pd
import pytest


"""
          Entry    Switch      From        To         Columns 15-24        Columns 25-34       Reference switch number
         number    number       bus       bus          (in seconds)         (in seconds)       (0 implies independence)
              1        22    X0001A    BRKR1A             0.012500             0.002405               0
              2        23    X0001B    BRKR1B             0.000000             0.002420              22
              3        24    X0001C    BRKR1C             0.016000             0.002420              22
"""

class TestStatisticalSwitchesList:
    @pytest.fixture(scope="class")
    def switches_list(self, example_lis_file):
        return ss.get_statistical_switches(example_lis_file)

    def test_keys(self, switches_list):
        expected_keys = sorted([
            "switch_number",
            "from_bus",
            "to_bus",
            "phase",
            "time_delay_center",
            "time_delay_sigma",
            "reference_switch_number",
        ])
        # Verify that all elements of the switches list have the right keys
        for s in switches_list:
            assert sorted(s.keys()) == expected_keys

    def test_count(self, switches_list):
        assert len(switches_list) == 3

    def test_ordering(self, switches_list):
        from_to_phase = [(v["from_bus"], v["to_bus"], v["phase"]) for v in switches_list]
        assert from_to_phase == [
            ("X0001", "BRKR1", "A"),
            ("X0001", "BRKR1", "B"),
            ("X0001", "BRKR1", "C"),
        ]

class TestStatisticalSwitchingResults:
    @pytest.fixture(scope="class")
    def simulation_results(self, example_lis_file):
        simulation_results_xr = ss.get_statistical_switching_results(example_lis_file)
        return simulation_results_xr

    def test_dimensions(self, simulation_results):
        assert set(simulation_results.dims) == {
            "simulation_number",
            "switch_number",
            "model_var",
        }

    def test_dimension_sizes(self, simulation_results):
        assert simulation_results.sizes["model_var"] == 124
        assert simulation_results.sizes["switch_number"] == 3
        assert simulation_results.sizes["simulation_number"] == 500

    def test_model_var_is_multiindex(self, simulation_results):
        idx = simulation_results.coords["model_var"].to_index()
        assert isinstance(idx, pd.MultiIndex)

    def test_model_var_coords_are_explicit(self, simulation_results):
        assert "model_var" in simulation_results.coords
        assert "from_node" in simulation_results.coords
        assert "to_node" in simulation_results.coords
        assert "phase" in simulation_results.coords
        assert "var_type" in simulation_results.coords


    def test_model_var_index_names(self, simulation_results):
        idx = simulation_results.coords["model_var"].to_index()
        assert idx.names == ["from_node", "to_node", "phase", "var_type"]


    def test_model_var_uniqueness(self, simulation_results):
        idx = simulation_results.coords["model_var"].to_index()
        assert idx.is_unique

    def test_phase_values(self, simulation_results):
        phases = set(simulation_results.coords["phase"].values)
        assert phases == {"A", "B", "C", ""}

    def test_var_type_values(self, simulation_results):
        var_types = set(simulation_results.coords["var_type"].values)
        assert var_types == {"vp", "ie"}

    def test_var_type_counts(self, simulation_results):
        idx = simulation_results.coords["model_var"].to_index()
        counts = idx.get_level_values("var_type").value_counts()

        assert counts["vp"] == 87
        assert counts["ie"] == 37

    def test_select_vp(self, simulation_results):
        vp = simulation_results.sel(var_type="vp")
        assert vp.sizes["model_var"] == 87


    def test_select_from_node(self, simulation_results):
        sub = simulation_results.sel(model_var=dict(from_node="BUS2"))
        assert sub.sizes["model_var"] > 0

    def test_select_specific_variable(self, simulation_results):
        sel = simulation_results.sel(from_node="BUS2", to_node="ZNOL2", phase="A", var_type="vp")
        assert "model_var" not in sel.sizes


    def test_maxima_shape(self, simulation_results):
        assert simulation_results["maxima"].dims == ("simulation_number", "model_var")

    def test_times_of_maxima_shape(self, simulation_results):
        assert simulation_results["times_of_maxima"].dims == ("simulation_number", "model_var")

    def test_switch_times_shape(self, simulation_results):
        assert simulation_results["switch_times"].dims == (
            "simulation_number",
            "switch_number",
        )

    def test_maxima_finite(self, simulation_results):
        assert np.isfinite(simulation_results["maxima"].values).all()

    def test_model_var_ordering_prefix(self, simulation_results):
        idx = simulation_results.coords["model_var"].to_index()
        assert idx[:3].tolist() == [
            ("BUS2", "ZNOL2", "A", "vp"),
            ("BUS2", "ZNOL2", "B", "vp"),
            ("BUS2", "ZNOL2", "C", "vp"),
        ]

    def test_spotcheck_values(self, simulation_results):
        reference_maxima = [
            [955286.615, 0.1135952E7, -0.118507E7],
            [776791.974, 923750.331, -976930.82],
        ]
        reference_time_of_maxima = [
        	[0.0172725, 0.021285, 0.0352565],
        	[0.032863, 0.021862, 0.035645],
        ]
        ref_nodes = ["GENOA", "NROCH"]
        ref_sim_number = 10
        ds = simulation_results.sel(to_node="", var_type="vp", simulation_number=10).unstack().sel(from_node=ref_nodes)
        assert np.allclose(ds["maxima"], reference_maxima)
        assert np.allclose(ds["times_of_maxima"], reference_time_of_maxima)
