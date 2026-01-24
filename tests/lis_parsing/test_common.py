import json
import pytest
import atp_utils.lis_parsing.common as lc

class TestLoggedVariableLayout:
    @pytest.fixture(scope="class")
    def logged_vars(self, example_lis_file):
        return lc.get_output_variables(example_lis_file)

    def test_structure(self, logged_vars):
        # Each entry is (from_node, to_node, phase, var_type)
        assert all(len(v) == 4 for v in logged_vars)

    def test_invariants(self, logged_vars):
        for from_node, to_node, phase, var_type in logged_vars:
            assert isinstance(from_node, str)
            assert isinstance(to_node, str)
            assert var_type in ("vp", "ie")

            if phase == "":
                continue

            assert phase in ("A", "B", "C")

    def test_counts_by_type(self, logged_vars):
        vp = [v for v in logged_vars if v[3] == "vp"]
        ie = [v for v in logged_vars if v[3] == "ie"]

        # Known from LIS layout
        assert len(vp) == 87
        assert len(ie) == 37

    def test_vp_ordering(self, logged_vars):
        vp = [v for v in logged_vars if v[3] == "vp"]

        assert vp[:3] == [
            ("BUS2", "ZNOL2", "A", "vp"),
            ("BUS2", "ZNOL2", "B", "vp"),
            ("BUS2", "ZNOL2", "C", "vp"),
        ]

        assert vp[-3:] == [
            ("BVCKW", "", "B", "vp"),
            ("BVCKW", "", "C", "vp"),
            ("BVCKW", "", "A", "vp"),
        ]

    def test_ie_ordering(self, logged_vars):
        ie = [v for v in logged_vars if v[3] == "ie"]

        assert ie[:3] == [
            ("BRKR1", "BUS1", "A", "ie"),
            ("BRKR1", "BUS1", "B", "ie"),
            ("BRKR1", "BUS1", "C", "ie"),
        ]

        assert ie[-3:] == [
            ("BUS2", "XL3Y", "B", "ie"),
            ("BUS2", "XL3Y", "C", "ie"),
            ("XL3Y", "TERRA", "", "ie"),
        ]

    def test_phase_validity(self, logged_vars):
        for from_node, to_node, phase, var_type in logged_vars:
            assert phase in {"A", "B", "C", ""}

    def test_uniqueness(self, logged_vars):
        assert len(set(logged_vars)) == len(logged_vars)

    def test_all_values(self, data_dir, logged_vars):
        f = (data_dir / "Example_with_statistic_switch-expected_vars.json").read_text()
        expected = json.loads(f)

        # expected file must now store 4-tuples
        expected = [tuple(v) for v in expected]

        assert logged_vars == expected
