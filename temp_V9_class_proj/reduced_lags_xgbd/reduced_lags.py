#### Importing Packages ####
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import nbinom
from scipy.stats import norm
import warnings
from xgboost_distribution import XGBDistribution
from scipy.stats import nbinom
import sys

############################# Functions ########################################
def quantile_score(y, y_quantiles, quantiles=np.arange(0.01, 1.0, 0.01)):
    """
    input:
        y = the true temperature value for a given month 
        y_quantiles = the distribution (quantile) values from .01 to .99 incramenting by .01 
        quantiles = the quantile sections (.01 to .99 incramenting by 0.01)
    
    Output: the quantile score 
    
    The purpose of this function is to calculate the quantile score for a given county and month (month by month level)
    
    Example:
    y = 8
    predicted_quantiles = [6,7,8,9,10]
    quantiles = [.2, .4, .6, .8, 1.0]
    
    Thus, the difference would be y - predicted_quantiles = [2, 1, 0, -1, -2]
    
    Hence for diff[ >= 0]
    (.2)(2) = .4
    (.4)(1) = .4
    (.6)(0) = 0
    
    For diff < 0:
    (.2)(|1|) = .2
    (0)(|-2|) = 0
    
    Avg = .10 / 5 = .2
    
    This is the quantile score for that month
    """
    diff = y - y_quantiles
    error = np.zeros(len(diff))
    error[diff >= 0] = quantiles[diff >= 0] * np.abs(diff[diff >= 0])
    error[diff < 0] = (1 - quantiles[diff < 0]) * np.abs(diff[diff < 0])
    return np.sum(error) / len(quantiles)

def create_lagged_variables(data, variables, lags, groupby_col = "county"):
    """
    Function creates lags of a given variable in a data frame. Accounting for specific county to avoid data leakage 
    
    Input: 
        data = data frame in question
        variables = list of variables we take the lag of 
        lags = number of lags we wanna take
    output:
        new column to data with the represented lag
    """
    for var in variables:
        for lag in lags:
            data[f"{var}_L{lag}"] = data.groupby(groupby_col)[var].shift(lag)
    
    return data
################################################################################
warnings.simplefilter(action = "ignore", category= pd.errors.PerformanceWarning)
warnings.simplefilter(action = "ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None # supress SettingWithCopyWarning
################################################################################

########################## Reading in Weather Data ###################################

data = pd.read_csv("/home/users/jelazaro/temp_forecasting/data/cleaned_weather_ds_v1.csv")

########### Prevent data Leakage ################
county_name = sys.argv[1]
data = data[data["county"] == county_name]
#################################################



prev_yr_lags = [1,6,12]

lagged_vars = ["temperature"]

data = create_lagged_variables(data, lagged_vars, prev_yr_lags)

### Applying the lags for soil_temperature
prev_yr_lags = [12]
lagged_vars = ["soil_temperature_layer_1"]

data = create_lagged_variables(data, lagged_vars, prev_yr_lags)
data = data.sort_values(by = ["year", "county", "month"])

### Create list of variables to fit the data ###
fit_vars = ["year", "month", 'temperature_L1', 'temperature_L6', 'temperature_L12', 'soil_temperature_layer_1_L12']
#### Now we go ahead and subset the data ####

data = data[data["year"] >= 1991]

################################ Model Validation ##############################

### Lists to store results ###
all_feature_importance = []

all_combined_predict_df_list = []

all_simulations_results = []


################ Counties: ['Fresno', 'Kern', 'Los Angeles', 'Merced', 'Orange', 'Placer', 'Riverside', 'Sacramento', 'San Bernardino', 'San Joaquin', 'Solano', 'Stanislaus', 'Tulare']

# start of the county loop

print(f"Training model for county: {county_name}")
county_data = data # recall that the data needs to be full to subset data 

X_train = {}
X_val = {}
y_train = {}
y_val = {}

best_eta = np.nan
best_score = np.inf 
best_model = {}

for eta in [0.01, 0.03, 0.05, 0.07, 0.1]:
    for max_depth in [3, 5, 7, 9]:
        for gamma in [0, 0.05, 0.1, 0.15, 0.2]:
            for min_child_weight in [1, 3, 5]:
                xgbd_model = XGBDistribution(
                    distribution="normal",
                    n_estimators=1000,
                    early_stopping_rounds=10,
                    eta=eta,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    gamma=gamma
                )
                model = {} # store ALL of the models information
                curr_score = 0.0

                for val_year in [2015, 2016, 2017]:
                    train_data = county_data[county_data["year"] < val_year]
                    val_data = county_data[county_data["year"] == val_year]

                    X_train[val_year] = np.array(train_data[fit_vars])
                    y_train[val_year] = np.array(train_data["temperature"])

                    X_val[val_year] = np.array(val_data[fit_vars])
                    y_val[val_year] = np.array(val_data["temperature"])

                    model[val_year] = xgbd_model.fit(X_train[val_year],
                                                     y_train[val_year],
                                                     eval_set = [(X_val[val_year],
                                                                  y_val[val_year])])
                    curr_score += model[val_year].best_score # gets the best score 
                    
                    

                # Model Scoring info (get the best model)
                if curr_score < best_score:
                    best_score = curr_score
                    best_eta = eta
                    best_model = model 
                    print(f'County: {county_name}, Curr best eta {eta}, max_depth {max_depth}, score {best_score}')


############################### Feature Importance #############################
feature_importance = np.mean([best_model[y].feature_importances_ for y in [2015, 2016, 2017]], axis=0)
nrounds = np.round(np.mean([best_model[y].best_iteration for y in [2015, 2016, 2017]]))


vars = [var for score, var in sorted(zip(feature_importance, fit_vars), reverse = True)]

# vars = []
# for score, var in sorted(zip(feature_importance, fit_vars), reverse = True):
#     vars.append(var) 

scores = sorted(feature_importance, reverse = True)

feature_importance_df = pd.DataFrame({
    "feature": vars,
    "importance": scores
})

feature_importance_df["county"] = county_name
all_feature_importance.append(feature_importance_df)

############################# Now that we have our hyp params, we re-train############################
X_train = {}
X_test = {}
y_train = {}
y_test = {}

xgbd_model = XGBDistribution(
    distribution = "normal",
    n_estimators = int(nrounds),
    eta = best_eta,
    max_depth = best_model[2016].max_depth,
    min_child_weight = best_model[2016].min_child_weight, 
    gamma = best_model[2016].gamma
)


model = {}

#### The idea is that now that we have ID'd our best Hyps, we go ahead and train the model from scratch using the best hyps

for test_year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]: # 6 years Mauricio wanted 
    train = county_data[county_data["year"] < test_year]
    test = county_data[county_data["year"] == test_year]

    X_train[test_year] = np.array(train[fit_vars])
    y_train[test_year] = np.array(train["temperature"])

    X_test[test_year] = np.array(test[fit_vars])
    y_test[test_year] = np.array(test["temperature"])

#    if test_year <= 2023:
#        best_model[test_year] = xgbd_model.fit(
#            X_train[test_year],
#            y_train[test_year],
#           eval_set = [(X_test[test_year], y_test[test_year])])

    best_model[test_year] = xgbd_model.fit(
        X_train[test_year],
        y_train[test_year])
        
        
######################### Using the models we trained and tested, we create simulations ############################
# predicting and scoring
predict_df = {}
dfs_to_combine = []

# Define the number of simulations at each trajectory
N_simulations = 10000

# Define the testing years
test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

for test_year in test_years:
    print(f"we are in test year: {test_year}")
    # use training data up to the year before the test year
    train = data[data["year"] < test_year]
    X_train[test_year] = np.array(train[fit_vars])
    y_train[test_year] = np.array(train["temperature"])

    # initialize the simulation results for the current year
    simulation_results = []

    # Initialize lagged values using actual data from Aug and later months
    cond = (county_data["year"] == test_year)
    initial_data = county_data.loc[cond].copy()

    for simulation in range(N_simulations):
        print(f"we are in simulation: {simulation}")
        # reset the lagged variables to the initial conditions for ea simulation
        simulated_data = initial_data.copy()

        # simulate starting from aug for all test years
        months = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        max_lag = 8 # only update lags up to L4

        for month in months:
            # extract vars of that month
            X_val = simulated_data[simulated_data["month"] == month][fit_vars]

            # use the best model to predict the dist pars
            params = best_model[test_year].predict(X_val) # I am unsure if this is correct. Should I pick from the validation set??
            mu = params.loc # mean
            sigma = params.scale # Standard Dev

            # Simulate temp for the current month using predicted dist
            simulated_temperature = np.random.normal(mu, sigma)

            # Store mu and sigma into the Df
            simulated_data.loc[simulated_data["month"] == month, "mu"] = mu
            simulated_data.loc[simulated_data["month"] == month, "sigma"] = sigma

            # store the simulated temp
            simulated_data.loc[simulated_data["month"] == month, "simulated_temperature"] = simulated_temperature
            simulated_data.loc[simulated_data["month"] == month, "simulation_number"] = simulation + 1
            
            # update the lagged variable up to L4
            for lag in range(1, max_lag + 1):
                target_month = month + lag
                if target_month <= 12:
                    lag_column_name = f"temperature_L{lag}"
                    simulated_data.loc[simulated_data["month"] == target_month, lag_column_name] = simulated_temperature

        simulation_results.append(simulated_data)
    combined_simulation_df = pd.concat(simulation_results, ignore_index = True)
    combined_simulation_df["year"] = test_year
    
    # combine all simulations for the current year into a single df
    combined_simulation_df["county"] = county_name

    columns_to_keep = ["county", "year", "month", "temperature", "simulated_temperature", "mu", "sigma", "simulation_number", "temperature_L12"]

    combined_simulation_df = combined_simulation_df[columns_to_keep]

    all_simulations_results.append(combined_simulation_df)

all_simulation_results_df = pd.concat(all_simulations_results, ignore_index = True)
###################### Calculating the Quantile Score #########################
all_simulation_results_df = all_simulation_results_df[["county", "year", "month", "temperature", "simulated_temperature", "mu", "sigma", "simulation_number"]]
all_simulation_results_df = all_simulation_results_df[all_simulation_results_df["month"] >= 1]

# intialize the list to collect results
results = []

# Define the quantile level
quantiles = np.arange(0.01, 1.0, 0.01)
quantile_labels = [f"q{round(q,2)}" for q in quantiles]

# get unique combinations of years and months
year_month_combinations = all_simulation_results_df[["year", "month"]].drop_duplicates() # gets every year with its respective month (ONCE) in our test set

for _, row in year_month_combinations.iterrows():
    test_year = row["year"]
    month = row["month"]
    print(f"processing year: {test_year}, month: {month}")

    # filter data for the current year and month
    cond = (all_simulation_results_df["year"] == test_year) & (all_simulation_results_df["month"] == month)

    # extract simulated temps
    sim_temp_np = all_simulation_results_df.loc[cond, "simulated_temperature"].to_numpy()

    # check if there are sim temp to process
    if len(sim_temp_np) > 0:
        # calculate ordered quantiles using np.quantile
        ordered_quantiles = np.quantile(sim_temp_np, quantiles)

        # prepare a dict to store the results
        result = {
            "year": test_year,
            "month": month
        }

        if test_year <= 2024: # change once we get 2024 data (will make sense once we introduce 2025)
            # retreive the observed temp for the month
            observed_temp_series = all_simulation_results_df.loc[cond, "temperature"]
            observed_temp = observed_temp_series.iloc[0]
            result["temperature"] = observed_temp

            # compute the quantile score once per month
            q_score = quantile_score(
                observed_temp,
                ordered_quantiles
            )

            result["quantile_score"] = q_score
        else:
            result["temperature"] = np.nan
            result["quantile_score"] = np.nan

    else:
        result["temperature"] = np.nan
        result["quantile_score"] = np.nan
    results.append(result)
else:
    print(f"no simulated data for year {test_year}, month {month}")

quantile_scores_df = pd.DataFrame(results)
quantile_scores_df["county"] = county_name
all_simulation_results_df = pd.concat(all_simulations_results, ignore_index = True)
all_feature_importance_df = pd.concat(all_feature_importance, ignore_index = True)


quantile_scores_df.to_csv(f"/scratch/users/jelazaro/temp_V9_results/red_lags_results_xgbd/quantile_scores/{county_name}_QS_reduced_lags.csv")
all_simulation_results_df.to_csv(f"/scratch/users/jelazaro/temp_V9_results/red_lags_results_xgbd/simulation_values/{county_name}_simulation_values_reduced_lags.csv")
all_feature_importance_df.to_csv(f"/scratch/users/jelazaro/temp_V9_results/red_lags_results_xgbd/ft_importance/{county_name}_ft_importance_reduced_lags")
