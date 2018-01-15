from ..etc import throughput


def test_etc():
    wavelengths = [3.5, 6]  # max and min
    trans = throughput(wavelengths)
    assert trans[0] > trans[1]