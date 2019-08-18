from src.feature_extraction import *
from src.util import *
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from const import *
import csv
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe, Trials, STATUS_OK, hp, fmin

from timeit import default_timer as timer
import ast
import warnings
warnings.filterwarnings("ignore")


def objective(hyperparameters):
    """
       Return an Interactive choropleth map of London boroughs with some of the top london attractions mark on the map

        Argument
       ========
       hyperparameters: dict, contains the model hyperparameters

       Returns
       ========
       dictionary with information for evaluation

       """
    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type and subsample to top levl keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples', 'max_depth']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    train_set = lgb.Dataset(train_features, label=train_labels, categorical_feature=CATEGORY_COLUMNS)

    start = timer()

    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold=N_FOLDS,
                        early_stopping_rounds=100, metrics='rmse',
                        categorical_feature=CATEGORY_COLUMNS,
                        stratified=False, seed=50)

    run_time = timer() - start
    # Extract the best score
    best_score = cv_results['rmse-mean'][-1]
    n_estimators = len(cv_results['rmse-mean'])

    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

    # write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([best_score, hyperparameters, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': best_score, 'hyperparameters': hyperparameters,
            'train_time': run_time, 'status': STATUS_OK}


def evaluate(results, name):
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
    print(f"The higest cross validation score from {name} was {new_results.loc[0, 'score']:.5f}"
          f" found on iteration {new_results.loc[0, 'iteration']}")

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

    return hyp_df


if __name__ == '__main__':
    # Create a file and open a connection
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)
    # Write column names
    writer.writerow(HEADERS)
    print("Writing file has finished......")
    of_connection.close()

    listing = pd.read_csv('../preprocessed_data/listings.csv')
    for col in CATEGORY_COLUMNS:
        listing.loc[:, col] = listing.loc[:, col].astype('category')
    price = pd.read_csv('../preprocessed_data/listings_price.csv', header=None, names=['price'])
    train_features, test_features, train_labels, test_labels = train_test_split(listing, price, test_size=0.2,
                                                                                random_state=42)

    space = {
        'boosting_type': hp.choice('boosting_type',
                                   [{'boosting_type': 'gbdt', 'subsample': hp.uniform(
                                       'gbdt_subsample', 0.5, 1)},
                                    {'boosting_type': 'dart', 'subsample': hp.uniform(
                                        'dart_subsample', 0.5, 1)},
                                    {'boosting_type': 'goss', 'subsample': 1.0}]),
        'num_leaves': hp.quniform('num_leaves', 30, 255, 2),
        'max_depth': hp.quniform('max_depth', 3, 12, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        'is_unbalance': hp.choice('is_unbalance', [True, False]),
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": -1
    }
    # record results
    trials = Trials()

    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials,
                max_evals=MAX_EVALS)
    print(f"Params after tuning: {best}")

    results = pd.read_csv(OUT_FILE)
    bayes_results = evaluate(results, name='BayesianFinal')




