"""Test Module."""

import flood_tool
import numpy as np

# from pytest import mark


tool = flood_tool.Tool()


def test_get_easting_northing_from_gps_lat_long():
    """test sets for get_easting_northing_from_gps_lat_long """

    data1 = flood_tool.get_easting_northing_from_gps_lat_long([55.5],
                                                              [-1.54])[0][0]
    data2 = flood_tool.get_easting_northing_from_gps_lat_long([55.5],
                                                              [-1.54])[1][0]

    if data1 is NotImplemented:
        assert False
    if data2 is NotImplemented:
        assert False

    assert np.isclose(data1, 429157.0).all()
    assert np.isclose(data2, 623009).all()


def test_get_gps_lat_long_from_easting_northing():
    """test sets for test_get_gps_lat_long_from_easting_northing """

    data1 = flood_tool.get_gps_lat_long_from_easting_northing([429157],
                                                              [623009])[0][0]
    data2 = flood_tool.get_gps_lat_long_from_easting_northing([429157],
                                                              [623009])[1][0]

    if data1 is NotImplemented:
        assert False
    if data2 is NotImplemented:
        assert False

    assert np.isclose(data1, 55.5).all()
    assert np.isclose(data2, -1.540008).all()


def test_get_easting_northing():
    """test sets for test_get_easting_northing """

    data = tool.get_easting_northing(['YO62 4LS'])

    if data is NotImplemented:
        assert False

    assert np.isclose(data.iloc[0].easting, 467631.0).all()
    assert np.isclose(data.iloc[0].northing, 472825.0).all()


# @mark.xfail  # We expect this test to fail until we write some code for it.
def test_get_lat_long():
    """test sets for test_get_lat_long """

    data = tool.get_lat_long(['YO62 4LS'])

    if data is NotImplemented:
        assert False

    assert np.isclose(data.iloc[0].Latitude, 54.147, 1.0e-3).all()
    assert np.isclose(data.iloc[0].Longitude, -0.966, 1.0e-3).all()


def test_set_postcodes():
    """test sets for set_postcodes"""
    assert tool.set_postcodes(['CH60 0wA']).values == 'CH600WA'
    assert tool.set_postcodes(['ak1 2A']).values == 'AK  12A'


if __name__ == "__main__":
    test_get_easting_northing()
    test_get_lat_long()
    test_set_postcodes()
    test_get_easting_northing_from_gps_lat_long()
    test_get_gps_lat_long_from_easting_northing()
