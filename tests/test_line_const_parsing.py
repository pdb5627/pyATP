import pytest
from math import pi
from pytest import approx
from atp_utils import LineConstPCHCards


@pytest.fixture(scope="class")
def example_pch_file1(data_dir):
    path = data_dir / "NB1_ATP_SHIELD5.pch"
    lines = path.read_text(encoding="iso-8859-1").splitlines()
    return lines

@pytest.fixture(scope="class")
def line_const_pch_cards1(example_pch_file1):
    line_params = LineConstPCHCards()
    line_params.read(example_pch_file1)
    return line_params


class TestLineConstPCHCards1:
    def test_shape(self, line_const_pch_cards1):
        assert line_const_pch_cards1.Z.shape == (4, 4)
        assert line_const_pch_cards1.Y.shape == (4, 4)

    def test_units(self, line_const_pch_cards1):
        assert line_const_pch_cards1.data["units"]["XOPT"] == approx(60.)
        assert line_const_pch_cards1.data["units"]["COPT"] == approx(60.)

    def test_symmetry(self, line_const_pch_cards1):
        assert line_const_pch_cards1.Z.T == approx(line_const_pch_cards1.Z)
        assert line_const_pch_cards1.Y.T == approx(line_const_pch_cards1.Y)

    def test_spot_check_values(self, line_const_pch_cards1):
        ω = 2*pi*60
        assert line_const_pch_cards1.Z[0, 0].real == approx(0.215265163)
        assert line_const_pch_cards1.Z[0, 0].imag == approx(1.36707181 / ω)
        assert line_const_pch_cards1.Y[0, 0].imag == approx(4.32577756 / ω / 1e6)

        assert line_const_pch_cards1.Z[3, 0].real == approx(0.0882048251)
        assert line_const_pch_cards1.Z[3, 0].imag == approx(0.513657388 / ω)
        assert line_const_pch_cards1.Y[3, 0].imag == approx(-0.652986076 / ω / 1e6)



@pytest.fixture(scope="class")
def example_pch_file2(data_dir):
    path = data_dir / "NB1_INTERNAL_SHIELD6.pch"
    lines = path.read_text(encoding="iso-8859-1").splitlines()
    return lines

@pytest.fixture(scope="class")
def line_const_pch_cards2(example_pch_file2):
    line_params = LineConstPCHCards()
    line_params.read(example_pch_file2)
    return line_params


class TestLineConstPCHCards2:
    def test_shape(self, line_const_pch_cards2):
        assert line_const_pch_cards2.Z.shape == (4, 4)
        assert line_const_pch_cards2.Y.shape == (4, 4)

    def test_units(self, line_const_pch_cards2):
        assert line_const_pch_cards2.data["units"]["XOPT"] == 0.
        assert line_const_pch_cards2.data["units"]["COPT"] == 0.

    def test_symmetry(self, line_const_pch_cards2):
        assert line_const_pch_cards2.Z.T == approx(line_const_pch_cards2.Z)
        assert line_const_pch_cards2.Y.T == approx(line_const_pch_cards2.Y)

    def test_atp_matches_internal(self, line_const_pch_cards1, line_const_pch_cards2):
        assert line_const_pch_cards1.Z.real == approx(line_const_pch_cards2.Z.real, rel=0.02)
        assert line_const_pch_cards1.Z.imag == approx(line_const_pch_cards2.Z.imag, rel=0.02)
        assert line_const_pch_cards1.Y.imag == approx(line_const_pch_cards2.Y.imag, rel=1e-5)
