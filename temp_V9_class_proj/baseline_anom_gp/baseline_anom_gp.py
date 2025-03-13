from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel, DotProduct
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


all_feature_importance = []

all_combined_predict_df_list = []

all_simulations_results = []

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
    

data = pd.read_csv("/home/users/jelazaro/temp_forecasting/data/cleaned_weather_ds_v1.csv")


########### Prevent data Leakage ################
county_name = sys.argv[1]
data = data[data["county"] == county_name]
#################################################

############ Lag variables #####################
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


lagged_vars = ["temp_anom"]
prev_yr_lags = [12]
data = create_lagged_variables(data, lagged_vars, prev_yr_lags)

data = data[data["year"] >= 1991]

data = data.sort_values(by = ["year", "county", "month"])

### Create list of variables to fit the data ###
#fit_vars = ["year", "month",'temp_anom_L12']

data = data[data["year"] >= 1991].copy()

scaler_year = StandardScaler()
data['year_normalized'] = scaler_year.fit_transform(data[['year']])

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

fit_vars = ["year_normalized", "month_sin", "month_cos", "temp_anom_L12"]

#################################################

np.random.seed(1)


######## setting up the GP params ###############
#kernel_mean = (
#    ConstantKernel(1.0, (1e-2, 1e2))
#    * ExpSineSquared(length_scale = 12, periodicity = 12, length_scale_bounds = (1e-2, 1e2))
#    + RBF(length_scale = 12, length_scale_bounds = (1e-2, 1e3)) 
#    + WhiteKernel(noise_level = 1e-3)
#)

#kernel_var = (
#    RBF(length_scale = 6, length_scale_bounds = (1e-2, 1e3))
#    + WhiteKernel(noise_level=1e-3)
#)
#################################################
kernel_trend = ConstantKernel(1.0, (1e-2, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
kernel_season = ConstantKernel(1.0, (1e-2, 1e2)) * ExpSineSquared(
    length_scale=3,             # or 12.0
    periodicity=12.0, 
    length_scale_bounds=(1e-2, 1e2),
    periodicity_bounds=(11, 13)
)
kernel_local = RBF(length_scale=50, length_scale_bounds=(1e-2, 1e3))
kernel_noise = WhiteKernel(noise_level=1e-3)

kernel_mean = kernel_season + kernel_local + kernel_trend + kernel_noise

kernel_var = (
    RBF(length_scale = 12, length_scale_bounds = (1e-2, 1e3))
    + WhiteKernel(noise_level=1e-3)
)


##################################################################
# 2.3. Validation to choose "best" GPs (optional)
##################################################################
# In your original code, you do a hyperparam search for XGB,
# here we'll do a simpler approach: we'll just define the kernels above
# and measure performance for val_year in [2015, 2016, 2017].
# We'll sum the RMSE across these years as "score."
# (You could do negative log-likelihood or something else.)
val_years = [2015, 2016, 2017]
total_rmse = 0.0

for val_year in val_years:
    train_data = data[data["year"] < val_year]
    val_data = data[data["year"] == val_year]

    # Prepare train arrays
    X_train = train_data[fit_vars].values
    y_train = train_data["temperature"].values

    # Fit GP for mean
    gp_mean = GaussianProcessRegressor(kernel=kernel_mean, alpha=1e-4, normalize_y=False, n_restarts_optimizer=2)
    gp_mean.fit(X_train, y_train)

    # Residuals for variance GP
    y_pred_train_mean = gp_mean.predict(X_train)
    residuals = y_train - y_pred_train_mean
    log_resid_sq = np.log(residuals**2 + 1e-6)

    # Fit GP for variance
    gp_var = GaussianProcessRegressor(kernel=kernel_var, alpha=1e-6, normalize_y=False, n_restarts_optimizer=2)
    gp_var.fit(X_train, log_resid_sq)

    # Evaluate on val_data
    X_val = val_data[fit_vars].values
    y_val = val_data["temperature"].values

    y_val_mean = gp_mean.predict(X_val)
    log_val_var = gp_var.predict(X_val)
    val_sigma = np.sqrt(np.exp(log_val_var))  # shape (n_samples,)

    # We'll just compute RMSE on the mean for "score"
    val_rmse = np.sqrt(np.mean((y_val_mean - y_val)**2))
    total_rmse += val_rmse

print(f"[Validation] Sum of RMSE over {val_years} = {total_rmse:.3f}")
# If you wanted multiple kernel combos, you'd do it in a loop, pick best, etc.
# We'll just keep the single kernel approach for simplicity.

##################################################################
# 2.4. Final "Loop" Over Test Years
##################################################################
# Now that we've "decided" on the kernel approach, let's do exactly
# what your original code does: for test_year in [2018..2024], re-fit
# on data < test_year, then simulate month-by-month.
# Assume quantiles, columns_to_keep, and fit_vars are defined, as well as the functions above.
columns_to_keep = ["county", "year", "month", "temperature", "mu", "sigma", "temp_anom_L12", "quantile_score"]
quantiles = np.array([round(q, 2) for q in np.arange(0.01, 1, 0.01)])
predict_df = {}

for test_year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    print(f"\n=== Processing Test Year = {test_year} for Quantile Scoring ===")
    
    # Define test subset and re-fit GP models on training data < test_year
    cond = (data["year"] == test_year)
    test_data = data.loc[cond].copy()
    
    # Prepare training data
    train_data = data[data["year"] < test_year]
    X_train = train_data[fit_vars].values
    y_train = train_data["temperature"].values
    
    # Fit GP for mean
    gp_mean = GaussianProcessRegressor(kernel=kernel_mean, alpha=1e-6, normalize_y=False, n_restarts_optimizer=2)
    gp_mean.fit(X_train, y_train)
    
    # Compute training residuals and fit GP for variance
    y_train_pred = gp_mean.predict(X_train)
    residuals = y_train - y_train_pred
    log_resid_sq = np.log(residuals**2 + 1e-6)
    gp_var = GaussianProcessRegressor(kernel=kernel_var, alpha=1e-6, normalize_y=False, n_restarts_optimizer=2)
    gp_var.fit(X_train, log_resid_sq)
    
    # Predict on test data
    X_test = test_data[fit_vars].values
    mu = gp_mean.predict(X_test)
    log_v = gp_var.predict(X_test)
    sigma = np.sqrt(np.exp(log_v))
    
    # Create a DataFrame with the GP parameters
    predict_df[test_year] = pd.DataFrame({
        "county": test_data["county"],
        "month": test_data["month"],
        "year": test_year,
        "temperature": test_data["temperature"],
        "temp_anom_L12": test_data["temp_anom_L12"],
        "mu": mu,
        "sigma": sigma
    })
    
    # Compute quantiles using the GP predictions
    for q in quantiles:
        col_name = f"q{round(q, 2)}"
        predict_df[test_year][col_name] = norm.ppf(q, mu, sigma)
    
    # Calculate the quantile score using your provided function
    if test_year <= 2024:  # For test years where we calculate the score
        predict_df[test_year]["quantile_score"] = predict_df[test_year].apply(
            lambda row: quantile_score(
                row["temperature"],
                row[[f"q{round(q, 2)}" for q in quantiles]].values  # ensure values are passed as array
            ),
            axis=1
        )
    else:  # e.g., for future years like 2025, if applicable
        predict_df[test_year]["quantile_score"] = 0

    # Subset columns to keep only the ones needed
    predict_df[test_year] = predict_df[test_year][columns_to_keep]

# Combine results for all test years into a single DataFrame
combined_df = pd.concat(predict_df.values(), ignore_index=True)


combined_df.to_csv(f"/scratch/users/jelazaro/temp_V9_results/baseline_anom_gp/quantile_scores/{county_name}_QS_OG_lags.csv")
