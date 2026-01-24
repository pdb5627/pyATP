from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def example_lis_file(data_dir):
    path = data_dir / "Example_with_statistic_switch.lis"
    lines = path.read_text(encoding="iso-8859-1").splitlines()
    return lines
