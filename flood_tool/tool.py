# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable

"""
This package is a Flood Risk Prediction tool, includeing the following
dunctions:
A Class to interact with a postcode database file.
Convert between UK ordnance survey easting/northing coordinates and GPS
latitude & longitude
Provide at some classifiers(knn, rf, nn) for postcodes in England into a
ten class flood probability scale based on provided labelled samples.
Provide a regression tool for median house price for postcodes in England,
given sampled data.
Calculate a predicted risk for these postcodes.
"""

import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from .geo import *  # noqa
from . import data_preprocessing
import numpy as np
import pandas as pd
import doctest


__all__ = ['Tool',
           'knn_classifier',
           'rf_classifier',
           'nn_classifier']


def knn_classifier(x_train, y_train, data, y_data, update_parameter=False):
    """Create, tune and train a KNN classifier with the
    splitted data set (postcodes_samples.csv).

    Args:
        X_train (pandas Dataframe): the training part of the dataset
        y_train (pandas DataFrame or Serie): the response label of the
        training set data (pandas DataFrame): the complete original dataset
        y (pandas DataFrame): the response label for the complete original
        dataset update_parameter (boolean) : if true make a grid search cv
        to find the best parameter

    Returns:
        sklearn.neighbors._classification.KNeighborsClassifier :
        a trained KNN classifier with the best parameter
    """

    # Hyperparameter Grid
    grid = {'n_neighbors': [1, 2, 3, 4, 5, 10, 20, 30, 50, 60]}
    # Instanciate Grid Search
    search = GridSearchCV(KNeighborsClassifier(), grid,
                          scoring='recall_macro',
                          cv=5,
                          n_jobs=-2  # paralellize computation
                          )

    if update_parameter:

        # Fit the model to find the best parameters
        search.fit(x_train, y_train)

        # Save the best parameter
        n_neigh = search.best_params_['n_neighbors']

        # Train a new model with the best parameter on the whole dataset
        opt_mod = KNeighborsClassifier(n_neighbors=n_neigh)

    else:
        opt_mod = KNeighborsClassifier(n_neighbors=10)

    opt_mod.fit(data, y_data)

    return opt_mod


def rf_classifier(x_train, y_train, data, y_data, update_parameter=False):
    """Create, tune and train a RandomForest classifier
    with the splitted data set (postcodes_samples.csv).

    Args:
        X_train (pandas Dataframe): the training part of the dataset
        y_train (pandas DataFrame or Serie): the response label of the
        training set data (pandas DataFrame): the complete original dataset
        y (pandas DataFrame): the response label for the complete original
        dataset update_parameter (boolean) : if true, make a gid search cv
        to tune the model

    Returns:
        sklearn.ensemble._forest.RandomForestClassifier :
        a trained RF classifier with the best parameter
    """

    # Hyperparameter Grid
    grid = {'n_estimators': [50, 100],
            'max_depth': [2, 10, 20]}
    # Instanciate Grid Search
    search = GridSearchCV(RandomForestClassifier(), grid,
                          scoring='recall_macro',
                          cv=5,
                          n_jobs=-1  # paralellize computation
                          )

    if update_parameter:

        # Fit the model to find the best parameters
        search.fit(x_train, y_train)

        # Save the best parameter
        n_tree = search.best_params_['n_estimators']
        maxdepth = search.best_params_['max_depth']

        # Train a new model with the best parameter on the whole dataset
        opt_mod = RandomForestClassifier(
            n_estimators=n_tree, max_depth=maxdepth)
    else:
        opt_mod = RandomForestClassifier(n_estimators=100, max_depth=25)

    opt_mod.fit(data, y_data)

    return opt_mod


def nn_classifier(x_train, y_train, data, y_data, update_parameter=False):
    """Create, tune and train a neural network multi layer perceptron
    classifier with the splitted data set (postcodes_samples.csv).

    Args:
        X_train (pandas Dataframe): the training part of the dataset
        y_train (pandas DataFrame or Serie): the response label of the
        training set data (pandas DataFrame): the complete original dataset
        y (pandas DataFrame): the response label for the complete original
        dataset update_parameter (boolean) : if true make a grid search cv
        to find the best parameter

    Returns:
        sklearn.neighbors._classification.KNeighborsClassifier :
        a trained KNN classifier with the best parameter
    """

    grid = {'hidden_layer_sizes': [(20, 30, 20), (10, 50, 10), (10, 10, 10)],
            'activation': ['relu', 'logistic'],
            'solver': ['lgbs', 'sgd', 'adam']}

    # Instanciate Grid Search
    search = GridSearchCV(MLPClassifier(), grid,
                          scoring='recall_macro',
                          cv=5,
                          n_jobs=-1  # paralellize computation
                          )

    if update_parameter is True:

        # Fit the model to find the best parameters
        search.fit(x_train, y_train)

        # Save the best parameter
        layers = search.best_params_['hidden_layer_sizes']
        sol = search.best_params_['solver']
        act = search.best_params_['activation']

        # Train a new model with the best parameter on the whole dataset
        opt_mod = MLPClassifier(hidden_layer_sizes=layers, activation=act,
                                solver=sol)

    else:
        opt_mod = MLPClassifier(n_neighbors=50)

    opt_mod.fit(data, y_data)

    return opt_mod


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='', sample_labels='',
                 household_file='', balanced_data='', stations='',
                 rainfall_wet='', rainfall_typ=''):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        sample_labels : str, optional
            Filename of a .csv file containing sample data on property
            values and flood risk labels.

        household_file : str, optional
            Filename of a .csv file containing information on households
            by postcode.
        """

        if postcode_file == '':
            postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_unlabelled.csv'))

        if sample_labels == '':
            sample_labels = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_sampled.csv'))

        if household_file == '':
            household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          'households_per_sector.csv'))

        if balanced_data == '':
            balanced_data = os.sep.join((os.path.dirname(__file__),
                                         'classification',
                                         'balanced_data.csv'))

        if stations == '':
            stations = os.sep.join((os.path.dirname(__file__),
                                    'resources',
                                    'stations.csv'))

        if rainfall_wet == '':
            rainfall_wet = os.sep.join((os.path.dirname(__file__),
                                        'resources',
                                        'wet_day.csv'))

        if rainfall_typ == '':
            rainfall_typ = os.sep.join((os.path.dirname(__file__),
                                        'resources',
                                        'typical_day.csv'))

        self.postcode_file = pd.read_csv(postcode_file)
        self.sample_lables = pd.read_csv(sample_labels)
        self.household_file = pd.read_csv(household_file)
        self.balanced_data = pd.read_csv(balanced_data)
        self.stations = pd.read_csv(stations)
        self.rainfall_wet = pd.read_csv(rainfall_wet)
        self.rainfall_typ = pd.read_csv(rainfall_typ)

    def set_postcodes(self, postcodes):
        """Cleanins the sequence of input postcodes
        from typos and human errors

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.core.series.Series
            Series containing cleaned postcodes. Invalid postcodes (i.e. not
            in the input unlabelled postcodes file) return as NaN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.set_postcodes(['cm13 2A N', 'T n 12 6 Y T', 'ls2 8lj', \
            'hU 17 9r H'])
        0    CM132AN
        1    TN126YT
        2    LS2 8LJ
        3    HU179RH
        Name: postcode, dtype: object
        >>> tool.set_postcodes(['IP1 5AH', 'DH8 7NE', 'DY5 3NN', 'HU3 1GB'])
        0    IP1 5AH
        1    DH8 7NE
        2    DY5 3NN
        3    HU3 1GB
        Name: postcode, dtype: object
        """
        postcode_df = pd.DataFrame({'postcode': postcodes})
        postcode_df['postcode'] = postcode_df['postcode'].str.upper()

        # If length is not 7 get rid of spaces. This
        # fixes e.g. "SW19 2AZ" -> "SW192AZ"
        postcode_df['postcode'] = postcode_df['postcode'].where(
            postcode_df['postcode'].str.len() == 7, postcode_df[
                'postcode'].str.replace(" ", ""))

        # If length is 5 (e.g. "W67HZ") add
        # two spaces in the middle (-> "W6  7HZ")
        postcode_df['postcode'] = postcode_df['postcode'].where(
            postcode_df['postcode'].str.len() != 5,
            postcode_df['postcode'].str[:2] + "  " + postcode_df[
                'postcode'].str[2:])

        # If length is 6 (e.g. "SW72AZ") add a space in
        # the middle and end(-> "SW7 2AZ")
        postcode_df['postcode'] = postcode_df['postcode'].where(
            postcode_df['postcode'].str.len() != 6,
            postcode_df['postcode'].str[:3] + " " + postcode_df[
                'postcode'].str[3:])

        return postcode_df['postcode']

    def get_easting_northing(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NaN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.get_easting_northing(['ch6 0 0wA', 'TQ12 6WF', \
            'S   S7 5  Q   a'])
                   easting  northing
        postcode
        CH600WA   326913.0  381988.0
        TQ126WF   281468.0   74769.0
        SS7 5QA   577512.0  188009.0
        >>> tool.get_easting_northing(['PL14 6SD', 'EX35 6HU', 'NE10 8NX', \
            'HR1 2JT', 'NE65 9DZ'])
                   easting  northing
        postcode
        PL146SD   221409.0   73664.0
        EX356HU   271919.0  149535.0
        NE108NX   428923.0  560646.0
        HR1 2JT   351754.0  239791.0
        NE659DZ   416722.0  603595.0
         """

        df1 = self.postcode_file
        df2 = self.sample_lables
        postcode_df = pd.concat([df1, df2], ignore_index=True, sort=False)
        postcode_df = postcode_df.fillna('np.nan')
        data = postcode_df[['postcode', 'easting', 'northing']]
        data = data.drop_duplicates()
        postcode_df = data.set_index(self.set_postcodes(data['postcode']))
        # postcode_df = data.set_index(data['postcode'])
        postcodes = self.set_postcodes(postcodes)
        postcode_df = postcode_df.drop(['postcode'], axis=1)
        cols = {'easting': postcode_df['easting'],
                'northing': postcode_df['northing']}
        en_df = pd.DataFrame(data=cols, index=postcodes)
        return en_df

    def get_lat_long(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NAN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.get_lat_long(['p O 3 5QP', 'IG10 2QH', 'tS 21 3BT'])
                   Latitude  Longitude
        postcode
        PO3 5QP   50.821343  -1.052293
        IG102QH   51.658602   0.068585
        TS213BT   54.655084  -1.450910
        >>> tool.get_lat_long(['LN11 7SW', 'BS30 5RY', 'HR3 5SX'])
                   Latitude  Longitude
        postcode
        LN117SW   53.384302   0.161671
        BS305RY   51.449035  -2.400703
        HR3 5SX   52.095724  -3.039906
        """

        postcodes = self.set_postcodes(postcodes)
        df = self.get_easting_northing(postcodes)
        latlong_df = \
            pd.DataFrame(df.apply(
                lambda x: get_gps_lat_long_from_easting_northing(  # noqa
                    x['easting'], x['northing']), axis=1))
        latlong_df['Latitude'], latlong_df['Longitude'] = zip(*latlong_df[0])
        latlong_df = latlong_df[['Latitude', 'Longitude']]
        latlong_df['Latitude'] = latlong_df['Latitude'].str.get(0)
        latlong_df['Longitude'] = latlong_df['Longitude'].str.get(0)
        return latlong_df

    @staticmethod
    def get_flood_class_methods():
        """
        Get a dictionary of available flood probablity classification methods.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_probability method.
        """
        dict_class_methods = {"KNN": 1, 'RandomForest': 2, 'Neural Network': 3}

        return dict_class_methods

    def get_flood_class(self, postcodes, method=0, update=False):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_probability_methods) the classification
            method to be used.
        update : boolean (optional)
            If set to True, tune the model with a GridsearchCV method.
            If False, take the implemented values for the parameters.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """

        # Import the balanced data train and add the rainfall as features
        postcodes = self.set_postcodes(postcodes)
        data = self.balanced_data
        data = data.dropna()

        # Separate the features and the response columns
        y_res = data.riskLabel
        data = data[['easting', 'northing']]

        # split the data
        x_train, x_test, y_train, y_test = train_test_split(data,
                                                            y_res,
                                                            test_size=0.3,
                                                            random_state=42)

        # Scaling the data for the Knn classifier
        sc = RobustScaler()
        sc.fit(x_train)
        X_train = sc.transform(x_train)
        X_test = sc.transform(x_test)  # noqa

        # Process the data test to have the same structure as the data train
        data_test = self.get_easting_northing(postcodes)
        data_test_sc = sc.transform(data_test)

        if method == 1:
            model = knn_classifier(x_train=X_train,
                                   y_train=y_train,
                                   data=data,
                                   y_data=y_res,
                                   update_parameter=update)
            predictions = model.predict(data_test_sc)
            data_test['flood_risk'] = predictions

            return data_test

        if method == 2:
            model = rf_classifier(x_train=x_train,
                                  y_train=y_train,
                                  data=data,
                                  y_data=y_res,
                                  update_parameter=update)
            predictions = model.predict(data_test)
            data_test['flood_risk'] = predictions

            return data_test

        if method == 3:
            model = nn_classifier(x_train=x_train,
                                  y_train=y_train,
                                  data=data,
                                  y_data=y_res)
            predictions = model.predict(data_test)
            data_test['flood_risk'] = predictions

            return data_test

        else:
            return pd.Series(data=np.ones(len(postcodes), int),
                             index=np.asarray(postcodes),
                             name='riskLabel')

    @staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """
        return {'all_england_median': 0, 'knn_regression': 1}

    def house_price_KNN_training(self):
        """
        This function fits and trains KNNRegressor in new data is inputted.
        If the data is not changed, then get_median_house_price_estimate
        function will run on pre-defined hyperparameters for computational
        efficiency.

        Parameters
        ----------

        pandas.core.frame.DataFrame
        DataFrame containing easting and northing and Median House Prices
        for model training.

        Returns
        -------
        KNeighborsRegressor model with assigned n_neighbors and weights
        hyperparameters based on GridSearch.
        """
        postcode_df = self.sample_lables
        postcode_df = postcode_df.dropna()
        postcode_df = postcode_df[postcode_df.medianPrice != 0]
        postcode_df = postcode_df[(postcode_df['medianPrice'] > 113100) & (
            postcode_df['medianPrice'] < 1207200)]
        X = postcode_df[['easting', 'northing']]
        y = postcode_df['medianPrice']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.3)
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                  15, 16, 17, 18, 19, 20],
                  'weights': ['uniform', 'distance']}
        knn = KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)
        model = model.fit(X_train, y_train)
        best_params = model.best_params_

        model = KNeighborsRegressor(
            n_neighbors=best_params['n_neighbors'], p=2,
            weights=best_params['weights'])
        final_model = model.fit(X, y)
        return final_model

    def house_price_estimate_en(self, en):
        """
        Function to be used in get_total_value, taking easting and northing
        values.
        Parameters
        ----------

        pandas.core.frame.DataFrame of only easting and northing values

        Returns
        -------

        numpy.ndarray of predicted values of Median House Price Estimates
        """
        # The n_neighbors is based on least root mean square error
        postcode_df = self.sample_lables
        X = postcode_df[['easting', 'northing']]
        y = postcode_df['medianPrice']
        model = KNeighborsRegressor(
            n_neighbors=6, weights='distance').fit(X, y)
        pred = model.predict(en)

        return pred

    def get_median_house_price_estimate(self, postcodes, method=0,
                                        training=False):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.get_median_house_price_estimate(['tr  13 8 e g', \
            'nn 14 4lU', 'y O1 1   rD'], 1)
                  Median House Price Prediction
        Postcode
        TR138EG                   482144.398964
        NN144LU                   366709.156072
        YO1 1RD                   358019.270575
        >>> tool.get_median_house_price_estimate(['dE 2  3DA', 'YO62 4LS', \
            'Nn 1 4 4  lU'], 1)
                  Median House Price Prediction
        Postcode
        DE2 3DA                   166138.586162
        YO624LS                   494297.375989
        NN144LU                   366709.156072
        """
        postcodes = self.set_postcodes(postcodes)

        if method == 1:
            if training is False:
                # cleaning the input postcodes
                postcodes = self.set_postcodes(postcodes)
                # creating a model
                model = KNeighborsRegressor(n_neighbors=6, weights='distance')
                postcode_df = self.sample_lables
                postcode_df = postcode_df.dropna()
                postcode_df = postcode_df[postcode_df.medianPrice != 0]
                postcode_df = postcode_df[
                    (postcode_df['medianPrice'] > 113100) & (
                        postcode_df['medianPrice'] < 1207200)
                ]
                xdata = postcode_df[['easting', 'northing']]
                ydata = postcode_df['medianPrice']
                model.fit(xdata, ydata)
                # using pre-written function to filter the dataset
                subset = self.get_easting_northing(postcodes)
                pred = model.predict(subset)
                line = {'Postcode': postcodes,
                        'Median House Price Prediction': pred}
                datares = pd.DataFrame(data=line).set_index('Postcode')
                return datares
            elif training is True:
                postcodes = self.set_postcodes(postcodes)
                model = self.house_price_KNN_training()
                subset = self.get_easting_northing(postcodes)
                pred = model.predict(subset)
                line = {'Postcode': postcodes,
                        'Median House Price Prediction': pred}
                datares = pd.DataFrame(data=line).set_index('Postcode')
                return datares

        else:
            return pd.Series(data=np.full(len(postcodes), 245000.0),
                             index=np.asarray(postcodes),
                             name='medianPrice')

    def get_total_value(self, locations):
        """
        Return a series of estimates of the total property values
        of a collection of postcode units or sectors.


        Parameters
        ----------

        locations : sequence of strs
            Sequence of postcode units or sectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        # setting files and removing spaces in sectors
        household_file = self.household_file
        household_file["postcode sector"] = household_file["postcode sector"].\
            str.replace(" ", "")
        postcode_df = self.sample_lables
        postcode_df = postcode_df.dropna()
        postcode_df["sector"] = postcode_df["sector"].str.replace(" ", "")
        unlabelled_data_sectors = postcode_df["sector"]
        unlabelled_data_grouped = postcode_df.groupby(
            ["sector"])[["easting", "northing"]].mean()

        estimates = []

        for i in range(0, len(locations)):

            # getting locations without spaces
            locations_no_spaces = locations[i].replace(" ", "")
            # checking if location is a sector
            if unlabelled_data_sectors[unlabelled_data_sectors.isin(
               [locations_no_spaces])].empty is False:

                en = unlabelled_data_grouped[[
                    "easting", "northing"]].loc[locations_no_spaces]
                cols = {'easting': en.loc['easting'],
                        'northing': en.loc['northing']}
                en_df = pd.DataFrame(data=cols, index=[0])

                estimate = self.house_price_estimate_en(en_df)
                # finding value of all properties in sector
                total_sector_value = estimate * \
                    household_file[household_file[
                        "postcode sector"] == locations_no_spaces][
                            "households"].values

                estimates.append(total_sector_value)
            # checking if location is a unit (sector +2 other characters)
            elif unlabelled_data_sectors[unlabelled_data_sectors.isin(
                 [locations_no_spaces[:len(
                     locations_no_spaces) - 2]])].empty is False:
                #
                unit_sector = locations_no_spaces[:len(
                    locations_no_spaces) - 2]
                en = unlabelled_data_grouped[[
                    "easting", "northing"]].loc[unit_sector]
                cols = {'easting': en.loc['easting'],
                        'northing': en.loc['northing']}
                en_df = pd.DataFrame(data=cols, index=[0])
                # finding total value of all properties in the unit
                unit_value = self.house_price_estimate_en(
                    en_df) * household_file[household_file[
                        "postcode sector"] == unit_sector][
                            "households"].values / \
                    household_file[household_file["postcode sector"] ==
                                   unit_sector][
                                       "number of postcode units"].values

                estimates.append(unit_value)

            else:
                # print("invalid postcode attempted")
                estimates.append(np.array([np.nan]))

            df = pd.DataFrame(pd.Series(locations), columns=["location"])
            df["total value"] = pd.Series(estimates)
            df = df.set_index("location")
            df = pd.concat((df[col].apply(pd.Series) for col in df), axis=1)
            df.columns = ['Total Value']
        return df

    def get_annual_flood_risk(self, postcodes,  risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.


        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood probability classifiers, as
            predicted by get_flood_probability.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        postcodes = self.set_postcodes(postcodes)
        prob_percentage = [0.1, 0.05, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5]
        probabilities = [x / 100 for x in prob_percentage]
        if risk_labels:

            annual_flood_risk = pd.DataFrame({'postcodes': postcodes,
                                              'risk_Label': risk_labels})
            estimates = self.get_total_value(postcodes)['Total Value']
            annual_flood_risk['risk_labels'] = annual_flood_risk[
                'risk_labels'].replace(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], probabilities)

            # annual_flood_risk['total_value'] = estimates
            annual_flood_risk['total_value'] = list(estimates)
            annual_flood_risk['total_risk'] = \
                0.5 * annual_flood_risk['flood_risk'] * annual_flood_risk[
                    'total value']
            annual_flood_risk.set_index('postcode')

        else:
            method = 1
            annual_flood_risk = pd.DataFrame({'postcodes': postcodes})
            estimates = self.get_total_value(postcodes)['Total Value']
            # annual_flood_risk['risk_labels'] = self.get_flood_class(
            #     postcodes, method).flood_risk
            annual_flood_risk['risk_labels'] = list(self.get_flood_class(
                postcodes, method).flood_risk)
            annual_flood_risk['risk_labels'] = annual_flood_risk[
                'risk_labels'].replace(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], probabilities)

            # annual_flood_risk['total value'] = estimates
            annual_flood_risk['total value'] = list(estimates)
            annual_flood_risk['total_risk'] = 0.5 * annual_flood_risk[
                'risk_labels'] * annual_flood_risk['total value']
            annual_flood_risk.set_index('postcodes')

        return (annual_flood_risk)

        # raise NotImplementedError

    def get_closest_stations(self, postcodes):
        """Find the closest stations for a series of postcodes

        Args:
            postcodes (pandas DataFrame or Series): a list of postcodes

        Returns:

            Pandas DataFrame: a table with the postcodes, their coordinates
            (latitude/longitude) and the nearest stations which measure the
            rainfall.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.get_closest_stations(['tS 21 3BT', 'SE23 3HW'])
                Latitude  Longitude Rain_stationReference
        postcode
        TS213BT   54.655084  -1.450910                032822
        SE233HW   51.437903  -0.053763              289102TP
        """

        # Add the coordinates for each postcodes
        data = self.get_lat_long(postcodes)

        # Import the station dataset
        stations = self.stations
        stations.dropna(inplace=True)

        # Filter to only have the rainfall station
        stations = stations.loc[stations.stationName == 'Rainfall station']
        stations.reset_index(inplace=True, drop=True)

        # Find the nearest stations
        kd = KDTree(stations[["latitude", "longitude"]].values,
                    metric='euclidean')
        indices = kd.query(data[["Latitude", "Longitude"]], k=1)[1]

        # Add the mean rainfall for the station
        data['Rain_stationReference'] = stations.iloc[list(
            indices.reshape(1, len(indices))[0]), 0].values

        return data

    def find_rainfall_average_per_postcodes(self, postcodes):
        """Gives the average rainfall of the closest station for given
        postcodes for a typical day and a wet day

        Args:
            data (DataFrame): a dataframe with at least the coordiantes
            (latitude, longitude) for every postcodes and the nearest stations
            which measures the rainfall

        Returns:
            Dataframe: A table with the mean rainfall for a typical and a wet
            day added for every postcode

        Examples
        --------

        """
        # Import the rainfall data for wet day and process it
        rainfall_wet = self.rainfall_wet
        rainfall_wet = data_preprocessing.get_rainfall_data_processed(
            rainfall_wet)

        # Import the rainfall data for tipycal_day and process it
        rainfall_typ = self.rainfall_typ
        rainfall_typ = data_preprocessing.get_rainfall_data_processed(
            rainfall_typ)

        data = self.get_closest_stations(postcodes)
        data.reset_index(inplace=True)

        rainfall_wet = rainfall_wet[rainfall_wet['stationReference'].isin(
            list(data.Rain_stationReference.values))]
        rainfall_typ = rainfall_typ[rainfall_typ['stationReference'].isin(
            list(data.Rain_stationReference.values))]

        rainfall_typ_mean = rainfall_typ[[
            'stationReference', 'value']].groupby(
            ['stationReference']).mean()
        rainfall_typ_mean['stationReference'] = rainfall_typ_mean.index
        rainfall_typ_mean.columns = ['typ_rainfall', 'Rain_stationReference']

        rainfall_wet_mean = rainfall_wet[[
            'stationReference', 'value']].groupby(
            ['stationReference']).mean()
        rainfall_wet_mean['stationReference'] = rainfall_wet_mean.index
        rainfall_wet_mean.columns = ['wet_rainfall', 'Rain_stationReference']

        rainfall_mean = rainfall_wet_mean.copy()
        rainfall_mean['typ_rainfall'] = rainfall_typ_mean['typ_rainfall']

        data_f = data.merge(rainfall_mean, how='inner',
                            on='Rain_stationReference')
        data_f.columns = ['postcode',
                          'Latitude',
                          'Longitude',
                          'stationReference',
                          'wet_rainfall',
                          'typ_rainfall']

        rainfall_typ = rainfall_typ.merge(data_f[[
            'postcode', 'stationReference']], how='inner',
                            on='stationReference')
        rainfall_wet = rainfall_wet.merge(data_f[[
            'postcode', 'stationReference']], how='inner',
                            on='stationReference')

        return data_f, rainfall_typ, rainfall_wet


# doctest.testmod(name='set_postcodes', verbose=True)
doctest.testmod(
    optionflags=doctest.NORMALIZE_WHITESPACE,
    verbose=True,
    report=False)
