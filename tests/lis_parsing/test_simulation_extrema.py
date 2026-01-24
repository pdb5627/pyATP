import atp_utils.lis_parsing.simulation_extrema as se
import numpy as np
import pandas as pd
import pytest
import xarray as xr

class TestSimulationExtremaStacked:
    def make_ds(self, value):
        return xr.Dataset(
            data_vars=dict(
                maxima=("x", [value])
            ),
            coords=dict(x=[0]),
        )
    def test_stacks_simulations(self, mocker, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"{i}.txt"
            f.write_text(str(i))
            files.append(f)

        def fake_get_simulation_extrema(lines):
            value = int(lines[0])
            return self.make_ds(value)

        mocker.patch(
            "atp_utils.lis_parsing.simulation_extrema.get_simulation_extrema",
            side_effect=fake_get_simulation_extrema,
        )

        ds = se.get_simulation_extrema_stacked(files)

        assert ds["maxima"].sel(simulation_idx=2).item() == 2

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            se.get_simulation_extrema_stacked([])




class TestSimulationExtrema:
    @pytest.fixture(scope="class")
    def simulation_extrema(self, example_lis_file):
        ds = se.get_simulation_extrema(example_lis_file)
        return ds

    def test_dimensions(self, simulation_extrema):
        assert set(simulation_extrema.dims) == {
            "model_var",
        }

    def test_dimension_sizes(self, simulation_extrema):
        assert simulation_extrema.sizes["model_var"] == 124

    def test_model_var_is_multiindex(self, simulation_extrema):
        idx = simulation_extrema.coords["model_var"].to_index()
        assert isinstance(idx, pd.MultiIndex)

    def test_model_var_coords_are_explicit(self, simulation_extrema):
        assert "model_var" in simulation_extrema.coords
        assert "from_node" in simulation_extrema.coords
        assert "to_node" in simulation_extrema.coords
        assert "phase" in simulation_extrema.coords
        assert "var_type" in simulation_extrema.coords


    def test_model_var_index_names(self, simulation_extrema):
        idx = simulation_extrema.coords["model_var"].to_index()
        assert idx.names == ["from_node", "to_node", "phase", "var_type"]


    def test_model_var_uniqueness(self, simulation_extrema):
        idx = simulation_extrema.coords["model_var"].to_index()
        assert idx.is_unique

    def test_phase_values(self, simulation_extrema):
        phases = set(simulation_extrema.coords["phase"].values)
        assert phases == {"A", "B", "C", ""}

    def test_var_type_values(self, simulation_extrema):
        var_types = set(simulation_extrema.coords["var_type"].values)
        assert var_types == {"vp", "ie"}

    def test_var_type_counts(self, simulation_extrema):
        idx = simulation_extrema.coords["model_var"].to_index()
        counts = idx.get_level_values("var_type").value_counts()

        assert counts["vp"] == 87
        assert counts["ie"] == 37

    def test_select_vp(self, simulation_extrema):
        vp = simulation_extrema.sel(var_type="vp")
        assert vp.sizes["model_var"] == 87


    def test_select_from_node(self, simulation_extrema):
        sub = simulation_extrema.sel(model_var=dict(from_node="BUS2"))
        assert sub.sizes["model_var"] > 0

    def test_select_specific_variable(self, simulation_extrema):
        sel = simulation_extrema.sel(from_node="BUS2", to_node="ZNOL2", phase="A", var_type="vp")
        assert "model_var" not in sel.sizes

    def test_maxima_shape(self, simulation_extrema):
        assert simulation_extrema["maxima"].dims == ("model_var",)

    def test_times_of_maxima_shape(self, simulation_extrema):
        assert simulation_extrema["times_of_maxima"].dims == ("model_var",)

    def test_maxima_finite(self, simulation_extrema):
        assert np.isfinite(simulation_extrema["maxima"].values).all()

    def test_minima_shape(self, simulation_extrema):
        assert simulation_extrema["minima"].dims == ("model_var",)

    def test_times_of_minima_shape(self, simulation_extrema):
        assert simulation_extrema["times_of_minima"].dims == ("model_var",)

    def test_minima_finite(self, simulation_extrema):
        assert np.isfinite(simulation_extrema["minima"].values).all()

    def test_model_var_ordering_prefix(self, simulation_extrema):
        idx = simulation_extrema.coords["model_var"].to_index()
        assert idx[:3].tolist() == [
            ("BUS2", "ZNOL2", "A", "vp"),
            ("BUS2", "ZNOL2", "B", "vp"),
            ("BUS2", "ZNOL2", "C", "vp"),
        ]

    @pytest.fixture(scope="class")
    def values_to_spotcheck(self, simulation_extrema):
        ref_nodes = ["GENOA", "NROCH"]
        ref_sim_number = 10
        ds = (simulation_extrema
            .sel(to_node="", var_type="vp")
            .unstack()
            .sel(from_node=ref_nodes)
        )
        return ds

    def test_spotcheck_maxima(self, values_to_spotcheck):
        reference_values = [
            [891103.728, 944310.209, 868386.305],
            [778650.539, 745309.772, 759675.116],
        ]
        assert np.allclose(values_to_spotcheck["maxima"], reference_values)

    def test_spotcheck_times_of_maxima(self, values_to_spotcheck):
        reference_values = [
            [0.0175250, 0.019972, 0.0448145],
            [0.0166705, 0.055311, 0.0451875],
        ]
        assert np.allclose(values_to_spotcheck["times_of_maxima"], reference_values)

    def test_spotcheck_minima(self, values_to_spotcheck):
        reference_values = [
            [-741663.01, -0.100183E7, -0.118567E7],
            [-736542.25, -799138.6, -958306.2],
        ]
        assert np.allclose(values_to_spotcheck["minima"], reference_values)

    def test_spotcheck_times_of_minima(self, values_to_spotcheck):
        reference_values = [
            [0.0406655, 0.0155495, 0.0354785],
            [0.0412275, 0.0303345, 0.035242],
        ]

        assert np.allclose(values_to_spotcheck["times_of_minima"], reference_values)
