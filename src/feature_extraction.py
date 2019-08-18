import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
pd.options.mode.chained_assignment = None

class AmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='amenities'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        X.loc[:, self.column] = X.loc[:, self.column].str.replace("[{}]", "").str.replace('"', "").str.strip()
        count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
        transformed_col = count_vectorizer.fit_transform(X[self.column])
        trans_col_to_df = pd.DataFrame(transformed_col.toarray(), columns=count_vectorizer.get_feature_names())
        trans_col_to_df = trans_col_to_df.drop('', axis=1)

        amenities_to_remove = []
        for col in trans_col_to_df.columns:
            if (trans_col_to_df[col].sum() / len(X) * 100) < 15:
                amenities_to_remove.append(col)

        trans_col_to_df.drop(amenities_to_remove, axis=1, inplace=True)

        return trans_col_to_df


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(f"The DataFrame does not include the columns: {cols_error}")


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes in a list of values as its argument
    def __init__(self, features):
        self.features = features

    def bin_cat(self, df, col, bins, labels, na_label='unknown'):
        """
        A Helper function takes a column, and segment continuous data into discrete bins with group of
        labels and fill the na value of the column according to na_label

        Argument
        ========
        df: pandas dataframe
        col: str, column of dataframe
        bins: int, the criteria to bin by
        labels: arrays, specify labels for the returned bins
        na_label: values to replace missing values in col

        Returns
        """

        if df.loc[:, col].dtype == 'object':
            df.loc[:, col] = df.loc[:, col].str[:-1].astype('float64')

        df.loc[:, col] = pd.cut(df.loc[:, col], bins=bins,
                                labels=labels, include_lowest=True).astype('str')
        df.loc[:, col].replace('nan', 'unknown', inplace=True)
        df.loc[:, col].fillna(na_label, inplace=True)

        return df

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):

        X = self.bin_cat(X, 'host_response_rate',
                         bins=[0, 50, 75, 90, 100],
                         labels=['0-49%', '50-74%', '75-89%', '90-100%'])

        X = self.bin_cat(X, 'review_scores_rating',
                         bins=[0, 80, 95, 100],
                         labels=['0-79', '80-94', '95-100'],
                         na_label='no reviews')

        X.property_type.replace({
            'Townhouse': 'House',
            'Serviced apartment': 'Apartment',
            'Condominium': 'Apartment',
            'Bed and breakfast': 'Guesthouse',
            'Loft': 'Apartment',
            'Guest suite': 'Guesthouse',
            'Hostel': 'Guesthouse',
            'Boutique hotel': 'Hotel',
            'Bungalow': 'House',
            'Cottage': 'House',
            'Aparthotel': 'Hotel',
            'Cottage': 'House',
            'Aparthotel': 'Hotel',
            'Villa': 'House',
            'Tiny house': 'House',
            'Chalet': 'House',
            'Earth house': 'House',
            'Dome house': 'House'
        }, inplace=True)

        X.loc[~X.property_type.isin(
            ['House', 'Apartment', 'Guesthouse', 'Hotel']), 'property_type'] = 'Other'

        df_amenities = AmenitiesTransformer().fit_transform(X)

        X = pd.concat([X, df_amenities], axis=1, join='inner')
        X = X.drop('amenities', axis=1)
        for feat in ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic']:
            X.loc[:, feat] = X.loc[:, feat].fillna(X.loc[:, feat].mode()[0])

        for col in self.features:
            X.loc[:, col] = X.loc[:, col].astype('category')

        X = X.replace({'f': 0, 't': 1})

        return X


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, price_features, room_features):
        self.price_features = price_features
        self.room_features = room_features

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, self.price_features] = X.loc[:, self.price_features].apply(
            lambda x: x.str[1:])
        X.loc[:, self.price_features] = X.loc[:, self.price_features].apply(
            lambda x: x.str.replace(",", '')).astype('float64')
        X.loc[:, self.price_features] = X.loc[:, self.price_features].fillna(value=0)

        X.loc[:, self.room_features] = X.loc[:, self.room_features].fillna(
            X.loc[:, self.room_features].median())

        X.loc[:, 'host_listings_count'] = X.loc[:, 'host_listings_count'].fillna(
            X.loc[:, 'host_listings_count'].median())

        return X