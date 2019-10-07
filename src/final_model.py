from src.util import *
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from const import *
from util import feature_importance_plot

import ast
import warnings
warnings.filterwarnings("ignore")


def evaluate(results, train_features, train_labels, test_features, test_labels, name):
    """
       Return an Interactive choropleth map of London boroughs with some of the top london attractions mark on the map

        Argument
       ========
       hyperparameters: dict, contains the model hyperparameters

       Returns
       ========
       dictionary with information for evaluation

       """

    new_results = results.copy()
    # string to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)

    # Sort with best values on top
    new_results = new_results.sort_values('score', ascending=True).reset_index(drop=True)

    # Print out cross validation high score
    print(f"The higest cross validation score from {name} was {new_results.loc[0, 'score']:.5f}")

    # Use best hyperparameters to create a model
    hyperparameters = new_results.loc[0, 'hyperparameters']
    model = LGBMRegressor(**hyperparameters)

    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict(test_features)
    rmse = (mean_squared_error(test_labels, preds)) ** 0.5
    r2score = r2_score(test_labels, preds)
    adj_r2 = 1 - (1 - r2score) * (test_features.shape[0] - 1) / (test_features.shape[0] - test_features.shape[1] - 1)
    print(f"RMSE from {name} on test data = {rmse:.5f}")
    print(f"R2 score from {name} on test data = {r2score:.5f}")
    print(f"Adjusted R2 score from {name} on test data = {adj_r2:.5f}")

    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns=list(new_results.loc[0, 'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluted
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index=[0]),
                               ignore_index=True)

    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']

    lightgbm_feature_imp = feature_importance_plot(train_features, model, "light_gbm_feature" )

    return hyp_df, lightgbm_feature_imp


def main():
    listing = pd.read_csv('../preprocessed_data/listings.csv')
    for col in CATEGORY_COLUMNS:
        listing.loc[:, col] = listing.loc[:, col].astype('category')
    price = pd.read_csv('../preprocessed_data/listings_price.csv', header=None, names=['price'])
    listing.drop("id", axis=1, inplace=True)
    train_features, test_features, train_labels, test_labels = train_test_split(listing, price, test_size=0.2,
                                                                                random_state=42)

    results = pd.read_csv(OUT_FILE)
    bayes_results, lightgbm_feat_imp = evaluate(results, train_features, train_labels,
                                                test_features, test_labels, name='BayesianFinal')


if __name__ == '__main__':
    main()