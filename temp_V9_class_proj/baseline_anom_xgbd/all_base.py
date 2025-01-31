import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import nbinom
from scipy.stats import norm
import warnings
from xgboost_distribution import XGBDistribution
from scipy.stats import nbinom
import sys

############################### Functions #################################

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
############################Warning Supresions##################################

warnings.simplefilter(action = "ignore", category= pd.errors.PerformanceWarning)
#warnings.simplefilter(action = "ignore", category=RuntimeWarning)
#pd.options.mode.chained_assignment = None # supress SettingWithCopyWarning
################################################################################

########################## Reading in Weather Data ###################################

data = pd.read_csv("/home/users/jelazaro/temp_forecasting/data/cleaned_weather_ds_v1.csv")


########### Prevent data Leakage ################
county_name = sys.argv[1]
data = data[data["county"] == county_name]
#################################################

############ Prepping the Anomolies #############

# Historical averages from data before 2016
data_temp_bef_2016 = data[data["year"] < 2016]
month_avg = (
    data_temp_bef_2016
    .groupby(["county","month"], as_index=False)["temperature"]
    .mean()
    .rename(columns={"temperature": "hist_temp_avg"})
)

# Merge and compute anomalies for data before 2016
data_temp_bef_2016 = pd.merge(data_temp_bef_2016, month_avg, on=["county", "month"])
data_temp_bef_2016["temp_anom"] = data_temp_bef_2016["temperature"] - data_temp_bef_2016["hist_temp_avg"]

# Merge and compute anomalies for data from 2016 onwards
data_temp_after_2016 = data[data["year"] >= 2016]
data_temp_after_2016 = pd.merge(data_temp_after_2016, month_avg, on=["county", "month"])
data_temp_after_2016["temp_anom"] = data_temp_after_2016["temperature"] - data_temp_after_2016["hist_temp_avg"]

# Finally, combine the two datasets needed
data = pd.concat([data_temp_bef_2016, data_temp_after_2016], ignore_index=True)

data = data.sort_values(by = ["county", "year", "month"])

########### Creating the Lag 12 Baseline Boded ############
lagged_vars = ["temp_anom"]
prev_yr_lags = [12]
data = create_lagged_variables(data, lagged_vars, prev_yr_lags)

### Create list of variables to fit the data ###
fit_vars = data.columns.to_list()

fit_vars.remove("year")
fit_vars.remove("county")
fit_vars.remove("temperature") # pretty sure we gotta remove temperature since that is what we are predicting
print(fit_vars)

################################ Model Validation ##############################

### Lists to store results ###
all_feature_importance = []

all_combined_predict_df_list = []

all_simulations_results = []


################ Counties: ['Fresno', 'Kern', 'Los Angeles', 'Merced', 'Orange', 'Placer', 'Riverside', 'Sacramento', 'San Bernardino', 'San Joaquin', 'Solano', 'Stanislaus', 'Tulare']


######################## Model Training ####################################
# start of the county loop

print(f"Training model for county: {county_name}")
county_data = data 

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
                
                ######################## Model Validation ####################################

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


########################### Model Testing #####################################
feature_importance = np.mean([best_model[y].feature_importances_ for y in [2015, 2016, 2017]], axis=0)
nrounds = np.round(np.mean([best_model[y].best_iteration for y in [2015, 2016, 2017]]))

vars = [var for score, var in sorted(zip(feature_importance, fit_vars), reverse = True)]
scores = sorted(feature_importance, reverse = True)


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

    if test_year <= 2023:
        best_model[test_year] = xgbd_model.fit(
            X_train[test_year],
            y_train[test_year],
            eval_set = [(X_test[test_year], y_test[test_year])])

    else: # year 2024 
        best_model[test_year] = xgbd_model.fit(X_train[test_year], y_train[test_year])


########################### Feature Importance #################################
#feature_importance = np.mean([best_model[y].feature_importances_ for y in [2015, 2016, 2017]], axis=0)
#nrounds = np.round(np.mean([best_model[y].best_iteration for y in [2015, 2016, 2017]]))

#vars = [var for score, var in sorted(zip(feature_importance, fit_vars), reverse = True)]
#scores = sorted(feature_importance, reverse = True)


# Save feature importance data to CSV
feature_importance_df = pd.DataFrame({
    'feature': vars,
    'importance': scores
})
feature_importance_df['county'] = county_name
all_feature_importance.append(feature_importance_df)

all_feature_importance
all_feature_importance_df = pd.concat(all_feature_importance, ignore_index=True)

################################################################################

columns_to_keep = ["county", "year", "month", "temperature", "mu", "sigma", "temp_anom_L12", "quantile_score"]
predict_df = {}
quantiles = np.array([round(q, 2) for q in np.arange(0.01, 1, 0.01)])

######################### Quantile Scores #####################################

for test_year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    cond = (data["year"] == test_year)
    X_val = data.loc[cond, fit_vars]
    params = best_model[test_year].predict(X_val)

    predict_df[test_year] = pd.DataFrame({
        "county": data.loc[cond, "county"],
        "month": data.loc[cond, "month"],
        "year": test_year,  # Include year here
        "temperature": data.loc[cond, "temperature"],
        "temp_anom_L12": data.loc[cond, "temp_anom_L12"],
        "sigma": params.scale,
        "mu": params.loc
    })

    # create col names with quantiles AND calculate the value that goes in that quantile
    for q in quantiles:
        predict_df[test_year][f"q{round(q, 2)}"] = norm.ppf(q, predict_df[test_year]["mu"], predict_df[test_year]["sigma"])
    
    # using the row, we calculate the quantile score
    if test_year <= 2023:
        predict_df[test_year]["quantile_score"] = predict_df[test_year].apply(
            lambda row: quantile_score(row["temperature"], row[[f'q{q}' for q in quantiles]]), axis=1)
    if test_year == 2024:
        predict_df[test_year]["quantile_score"] = 0

    # Subset columns for the specific DataFrame
    predict_df[test_year] = predict_df[test_year][columns_to_keep]
combined_df = pd.concat(predict_df.values(), ignore_index=True)


combined_df.to_csv(f"/scratch/users/jelazaro/temp_V9_results/baseline_results_xgbd_anom/quantile_scores/{county_name}_QS_baseline.csv")
all_feature_importance_df.to_csv(f"/scratch/users/jelazaro/temp_V9_results/baseline_results_xgbd_anom/ft_importance/{county_name}_ft_imp_baseline.csv")
