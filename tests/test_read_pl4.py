import pyATP.read_pl4 as read_pl4


def test_pl4_to_dataframe(data_dir):
    """ This is just a smoke test for now. """
    filename = data_dir / "GraingerEx16_9.pl4"
    df = read_pl4.pl4_to_dataframe(filename)

