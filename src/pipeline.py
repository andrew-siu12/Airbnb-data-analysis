from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from feature_extraction import *
from util import *

if __name__ == '__main__':
    listing_df = load_csv("../data/listings.csv.gz")
    price = listing_df.price
    price = price.apply(lambda x: x[1:])
    price = price.apply(lambda x: x.replace(",", '')).astype(
        'float64').replace(0.0, 0.01)
    price = np.log(price)

    feature_columns = ['id', 'host_response_rate', 'host_is_superhost', 'host_listings_count',
                       'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed',
                       'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
                       'amenities', 'security_deposit', 'cleaning_fee', 'guests_included',
                       'extra_people', 'minimum_nights', 'maximum_nights', 'availability_90',
                       'number_of_reviews', 'review_scores_rating', 'requires_license',
                       'instant_bookable', 'is_business_travel_ready', 'cancellation_policy',
                       'require_guest_profile_picture', 'require_guest_phone_verification']

    price_col = ['security_deposit', 'cleaning_fee', 'extra_people']
    room_col = ['bathrooms', 'bedrooms']
    numerical_transformer = Pipeline(steps=[
        ('num_tranform', NumericalTransformer(price_col, room_col))
    ])

    category_columns = ['host_response_rate', 'neighbourhood_cleansed', 'property_type',
                        'room_type', 'review_scores_rating', 'cancellation_policy']
    categorical_transformer = Pipeline(steps=[
        ('cat_transform', CategoricalTransformer(category_columns))
    ])

    clf = Pipeline(steps=[
        ('columnselect', ColumnSelector(feature_columns)),
        ('cat', categorical_transformer),
        ('num', numerical_transformer)
    ])

    processed_df = clf.fit_transform(listing_df)

    processed_df.to_csv("../preprocessed_data/listings.csv", index=False)
    price.to_csv("../preprocessed_data/listings_price.csv", index=False)
    print("Finished transforming Data.....")
    print('-'*30)