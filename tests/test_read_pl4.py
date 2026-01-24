from atp_utils import pl4_to_dataframe


def test_pl4_to_dataframe(data_dir):
    """ This is just a smoke test for now. """
    filename = data_dir / "GraingerEx16_9.pl4"
    df = pl4_to_dataframe(filename)
