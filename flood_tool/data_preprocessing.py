# pylint: disable=missing-module-docstring

import numpy as np

__all__ = ['get_rainfall_data_processed',
           'get_rainfall_per_hour_data',
           'get_level_data_processed',
           'get_level_data_aggregate',
           'average_interval']


def average_interval(value):
    """Function that compute the average value in case the mm (rainfall) or
    mASD (rivers) values
    are presented as 'XXX | XXX' .

    Args:
        value (string): the raw value presented in the data

    Returns:
        float: the average value and converted in float type
    """

    try:
        result = float(value)
    except ValueError:
        list_avg = value.split("|")
        list_avg = [float(x) for x in list_avg]
        result = np.mean(list_avg)

    return result


def get_rainfall_data_processed(data):
    """Filter the data to get only the rainfall measurements.
    And process them to get rid of outliers, missing values or duplicates.

    Args:
        data (DataFrame): the raw data

    Returns:
        (DataFrame): the processed data
    """

    # Filtering the dataset to have only the rainfall measurements
    data = data.loc[data.parameter == 'rainfall', :]

    # drop the duplicates
    data.drop_duplicates(inplace=True)

    # drop the NA in the column of interest
    data.dropna(subset=['value'], inplace=True)

    data['value'] = data['value'].apply(average_interval)

    return data


def get_rainfall_per_hour_data(data, stations):
    """Generate the sum of the rainfall per hour per station.
    Merge the stations informations (coordinates).
    Classify the stations acocridng to the amount of rainfall per hour.
    Btw 0-2 mm : slight
    Btw 2-4 mm : moderate
    Btw 4-50 mm : heavy
    Over 50 mm : violent

    Args:
        data (DataFrame): the processed data filtered with the rainfall
        mesurements stations (DataFrame): dataset with the coordinates
        for each stations

    Returns:
        DataFrame: The rainfall data set with stations coordinates and
        rainfall classification added.
    """

    data[['new_date_f', 'min', 'second']] = \
        data.dateTime.str.split(":", expand=True,)

    data_h = data.groupby(['stationReference', 'new_date_f']).sum()

    data_h['class_rain'] = 'slight'
    data_h.loc[data_h.value >= 2, 'class_rain'] = 'moderate'
    data_h.loc[data_h.value >= 4.0001, 'class_rain'] = 'heavy'
    data_h.loc[data_h.value >= 50.0001, 'class_rain'] = 'violent'

    data_h = data_h.merge(data, how='inner', on=['stationReference',
                                                 'new_date_f'])
    data_h = data_h.merge(stations, how='inner', on='stationReference')

    return data_h


def get_level_data_processed(data):
    """Filter the data to get only the rivers height measurements.
    And process them to get rid of outliers, missing values or duplicates.

    Args:
        data (DataFrame): the raw data

    Returns:
        (DataFrame): the processed data
    """

    # Filtering the dataset to have only the rainfall measurements
    data = data.loc[data.parameter == 'level', :]
    data = data.loc[data.unitName == 'mASD', :]

    # drop the duplicates
    data.drop_duplicates(inplace=True)

    # drop the NA in the column of interest
    data.dropna(subset=['value'], inplace=True)

    data['value'] = data['value'].apply(average_interval)

    # drop the single outliers (value > 214 000)
    data = data.loc[data.value <= 250, :]
    # we suppose that a increase of the height of 250m above
    # usual height is an outlier.

    return data


def get_level_data_aggregate(data, stations):
    """Compute for each stations the standard deviation and the mean of the
    rivers height evolution.
    Add the stations coordinates.

    Args:
        data (DataFrame): the processed rivers data set
        stations (DataFrame): the stations dataset with their coordinates

    Returns:
        DataFrame: the final data Frame with the aggregates for each stations.
    """

    result = data.groupby(['stationReference']).agg(['mean', 'std'])
    result = result.merge(stations, how='inner', on='stationReference')

    return result
