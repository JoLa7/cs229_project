#### Importing Packages ####
import os
import argparse
import numpy as np
import pandas as pd
import warnings
import optuna
import optunahub
from xgboost_distribution import XGBDistribution

############################# Functions ########################################
def create_lagged_variables(data, variables, lags, groupby_col="county"):
    """
    Function creates lags of a given variable in a data frame. Accounting for specific county to avoid data leakage

    Input:
        data = data frame in question
        variables = list of variables we take the lag of
        lags = number of lags we want to take
    output:
        new column to data with the represented lag
    """

    """
    Function creates lags of a given variable in a data frame, accounting for a specific county to avoid data leakage.

    Input: 
        data = DataFrame containing time-series data
        variables = List of variables to create lagged values for
        lags = List of lag periods (e.g., [1, 2, 3, 12] for 1-month, 2-month, 3-month, and 12-month lags)
        groupby_col = Column used for grouping data before creating lags (default is "county" to prevent leakage across counties)

    Output:
        DataFrame with new columns containing lagged values of the specified variables.
    """
    for var in variables:
        for lag in lags:
            data[f"{var}_L{lag}"] = data.groupby(groupby_col)[var].shift(lag)

    return data

def objective(trial):
    # Define hyperparameter search space
    eta = trial.suggest_float("eta", 0.01, 0.1)  # log=True # Learning rate
    max_depth = trial.suggest_int("max_depth", 3, 9)  # Depth of tree
    gamma = trial.suggest_float("gamma", 0.0, 0.2)  # Minimum loss reduction
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)  # Minimum sum of instance weight needed in a child
    # subsample = trial.suggest_float("subsample", 0.5, 1.0)  # Subsampling ratio
    # colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)  # Fraction of features used per tree
    # reg_lambda = trial.suggest_float("reg_lambda", 0.01, 15)

    # Create model with hyperparameters
    xgbd_model = XGBDistribution(
        distribution="normal",
        n_estimators=1000,
        early_stopping_rounds=10,
        eta=eta,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma
    )

    # Split dataset for training and validation
    validation_years = [2015, 2016, 2017]
    curr_score = 0.0
    best_iterations = []
    model = {}

    for val_year in validation_years:
        train_data = county_data[county_data["year"] < val_year]
        val_data = county_data[county_data["year"] == val_year]

        X_train = np.array(train_data[fit_vars])
        y_train = np.array(train_data["temperature"])

        X_val = np.array(val_data[fit_vars])
        y_val = np.array(val_data["temperature"])

        # Fit model
        model[val_year] = xgbd_model.fit(X_train, y_train,
                                         eval_set=[(X_val, y_val)])

        curr_score += model[val_year].best_score
        best_iterations.append(model[val_year].best_iteration)

    avg_best_iteration = int(np.mean(best_iterations))
    
    # Log trial results
    trial_results = {
        "trial_number": trial.number,
        "loss_value": curr_score,
        "best_iteration": avg_best_iteration,
        "eta": eta,
        "max_depth": max_depth,
        "gamma": gamma,
        "min_child_weight": min_child_weight
    }
    optuna_results.append(trial_results)

    return curr_score

################################################################################
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # supress SettingWithCopyWarning
################################################################################

############################### Argument Parser ################################
parser = argparse.ArgumentParser(description="Hyperparameter Tuning Arguments")

parser.add_argument("--input_file", type=str, required=True, help="Input CSV file")
parser.add_argument("--county", type=str, required=True, help="County Name")
parser.add_argument("--trails", type=int, required=True, help="Number of trails")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

####################### Reading in Weather Data ################################
data = pd.read_csv(args.input_file)

########### Prevent data Leakage ################
print(f'Running {args.county} County Hyperparameter Tuning ...')
data = data[data["county"] == args.county]

#################################################
prev_yr_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]
lagged_vars = ["temperature"]
data = create_lagged_variables(data, lagged_vars, prev_yr_lags)

### Applying the lags for soil_temperature
prev_yr_lags = [12]
lagged_vars = ["soil_temperature_layer_1"]

data = create_lagged_variables(data, lagged_vars, prev_yr_lags)
data = data.sort_values(by=["year", "county", "month"])

### Create list of variables to fit the data ###
fit_vars = ["year", "month", 'temperature_L1', 'temperature_L2', 'temperature_L3', 'temperature_L4', 'temperature_L5', 'temperature_L6', 'temperature_L7', 'temperature_L8', 'temperature_L9', 'temperature_L10', 'temperature_L11', 'temperature_L12', 'temperature_L24', 'temperature_L36', 'soil_temperature_layer_1_L12']

#### Subset the data ####
data = data[data["year"] >= 1991]
county_data = data
optuna_results = []

# Initialize bayesian optimization
study = optuna.create_study(direction="minimize",
                            sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
                            study_name="Temp_Forecasting_Opt",
                            load_if_exists=True)

# Run optimization with set number of trials
study.optimize(objective, n_trials=args.trails)

# Convert all results to DataFrame and save as CSV
county_filename = args.county.replace(" ", "_")
results_df = pd.DataFrame(optuna_results)
all_results_filename = os.path.join(args.output_dir, f'{county_filename}_optuna_trial_results.csv')
results_df.to_csv(all_results_filename, index=False)  # Save as CSV

# Obtain average best iteration
best_trial = study.best_trial
best_trail_info = results_df[results_df["trial_number"] == best_trial.number]
best_avg_iteration = best_trail_info["best_iteration"].values[0]

# Obtain best hyperparameters
best_params = {
    "trial_number": best_trial.number,
    "loss_value": best_trial.value,
    "best_iteration": best_avg_iteration,
    "params": best_trial.params
}

# Save best hyperparameters as CSV
best_params_df = pd.DataFrame([best_params])
best_results_filename = os.path.join(args.output_dir, f'{county_filename}_best_optuna_trial_results.csv')
best_params_df.to_csv(best_results_filename, index=False)
