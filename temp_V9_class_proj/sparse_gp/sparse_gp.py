############################################################
###  GaussProcess_Simulation.py
###  (Mirrors our XGBD code structure for month-by-month simulation)
############################################################

import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ExpSineSquared, ConstantKernel, DotProduct
)

############################# Functions ########################################
def quantile_score(y, y_quantiles, quantiles=np.arange(0.01, 1.0, 0.01)):
    """
    Same as XGBD code. Given a single observed value y and an array of
    predicted quantiles, compute the quantile score.
    """
    diff = y - y_quantiles
    error = np.zeros(len(diff))
    error[diff >= 0] = quantiles[diff >= 0] * np.abs(diff[diff >= 0])
    error[diff < 0]  = (1 - quantiles[diff < 0]) * np.abs(diff[diff < 0])
    return np.sum(error) / len(quantiles)

def create_lagged_variables(data, variables, lags, groupby_col="county"):
    """
    Same as XGBD code. Creates lagged features for each variable and lag,
    grouped by a column (e.g. county).
    """
    for var in variables:
        for lag in lags:
            data[f"{var}_L{lag}"] = data.groupby(groupby_col)[var].shift(lag)
    return data

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

########################## Reading in Weather Data ##############################
data = pd.read_csv("/home/users/jelazaro/temp_forecasting/data/cleaned_weather_ds_v1.csv")

####### 1) Prevent Data Leakage: Subset the County #######
county_name = sys.argv[1]
data = data[data["county"] == county_name].copy()

####### 2) Create Lagged Variables (Exact same as XGBD) #######
temperature_lags = [6,12,24]
data = create_lagged_variables(data, ["temperature"], temperature_lags)

other_cov_lags = [12]
data = create_lagged_variables(data, ["soil_temperature_layer_1", "leaf_area_index_hi_veg"], other_cov_lags)

data.sort_values(by=["year","county","month"], inplace=True)
data = data[data["year"] >= 1991].copy()

####### 3) Define Feature Set (Same as XGBD) #######
fit_vars = ["temperature_L6","temperature_L12", "temperature_L24",
    "soil_temperature_layer_1_L12", "leaf_area_index_hi_veg"
]
# Drop any rows missing lag data
data.dropna(subset=fit_vars, inplace=True)

print(f"Data shape after dropping NA lags: {data.shape}")

####### Define GP Kernels #######
kernel_trend  = ConstantKernel(1.0, (1e-2, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
kernel_season = ConstantKernel(1.0, (1e-2, 1e2)) * ExpSineSquared(
    length_scale=3,
    periodicity=12.0,
    length_scale_bounds=(1e-2,1e2)
)
kernel_local  = RBF(length_scale=50, length_scale_bounds=(1e-2,1e3))
kernel_noise  = WhiteKernel(noise_level=1e-3)

kernel_mean = kernel_season + kernel_local + kernel_trend + kernel_noise
kernel_var  = (
    RBF(length_scale=12, length_scale_bounds=(1e-2,1e3))
    + WhiteKernel(noise_level=1e-3)
)

####### 5) Year-by-year "final" training & simulation loop #######
test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
N_simulations = 10000

all_simulations_results = []

for test_year in test_years:
    print(f"\n=== Processing Test Year = {test_year} ===")
    # a) Prepare training and test data
    train_data = data[data["year"] < test_year].copy()
    test_data  = data[data["year"] == test_year].copy()

    # b) Fit GP for Mean
    X_train = train_data[fit_vars].values
    y_train = train_data["temperature"].values

    gp_mean = GaussianProcessRegressor(kernel=kernel_mean,
                                       alpha=1e-6,
                                       normalize_y=False,
                                       n_restarts_optimizer=10)
    gp_mean.fit(X_train, y_train)

    # c) Fit GP for Variance
    y_pred_train = gp_mean.predict(X_train)
    residuals    = y_train - y_pred_train
    log_resid_sq = np.log(residuals**2 + 1e-6)

    gp_var = GaussianProcessRegressor(kernel=kernel_var,
                                      alpha=1e-2,
                                      normalize_y=False,
                                      n_restarts_optimizer=10)
    gp_var.fit(X_train, log_resid_sq)

    # d) Simulation
    #    Mirror the XGBD code logic: we create "initial_data" for test_year,
    #    run month-by-month, update lags up to L8, etc.
    cond = (test_data["year"] == test_year)
    initial_data = test_data.loc[cond].copy()
    # If no rows found for that year, skip
    if initial_data.shape[0] == 0:
        print(f"No data found for test_year={test_year} in {county_name}. Skipping.")
        continue

    simulation_results = []
    months_to_simulate = [4,5,6,7,8,9,10,11,12]  # same as XGB
    max_lag = 8  # only update lags up to L8

    for simulation in range(N_simulations):
        # i) Copy the “initial_data” for this run
        simulated_data = initial_data.copy()

        # ii) For each month, predict distribution and draw from Normal
        for month in months_to_simulate:
            cond_m = (simulated_data["month"] == month)
            if not cond_m.any():
                continue

            # Prepare input for GP
            X_val = simulated_data.loc[cond_m, fit_vars].values
            mu     = gp_mean.predict(X_val)
            logVar = gp_var.predict(X_val)
            sigma  = np.sqrt(np.exp(logVar))

            # Sample from Normal
            sampled_temp = np.random.normal(mu, sigma)

            # Save these predictions / samples
            simulated_data.loc[cond_m, "mu"]    = mu
            simulated_data.loc[cond_m, "sigma"] = sigma
            simulated_data.loc[cond_m, "simulated_temperature"] = sampled_temp
            simulated_data.loc[cond_m, "simulation_number"]     = simulation + 1

            # iii) Update the next months' lag columns (up to L8) with the newly sampled value
            for lag in range(1, max_lag + 1):
                future_month = month + lag
                if future_month <= 12:
                    lag_col = f"temperature_L{lag}"
                    simulated_data.loc[simulated_data["month"] == future_month, lag_col] = sampled_temp

        # store the entire simulated year
        simulation_results.append(simulated_data)

    # e) Combine all simulations for this year
    combined_sim_df = pd.concat(simulation_results, ignore_index=True)
    combined_sim_df["county"] = county_name
    combined_sim_df["year"]   = test_year

    # f) Keep columns consistent with XGB code
    columns_to_keep = [
        "county","year","month","temperature",
        "simulated_temperature","mu","sigma","simulation_number",
        # keep one example of a lag if you want to debug, or omit
        #"temperature_L12"
    ]
    # Only keep columns that exist in combined_sim_df
    columns_to_keep = [c for c in columns_to_keep if c in combined_sim_df.columns]
    combined_sim_df = combined_sim_df[columns_to_keep]

    # g) Append to global results
    all_simulations_results.append(combined_sim_df)

# Combine for all test_years
all_simulation_results_df = pd.concat(all_simulations_results, ignore_index=True)

###################### 6) Calculate Empirical Quantiles & Score ##############
quantiles = np.arange(0.01, 1.0, 0.01)
quantile_labels = [f"q{round(q,2)}" for q in quantiles]

results = []
year_month_combos = (
    all_simulation_results_df[["year","month"]]
    .drop_duplicates()
    .sort_values(["year","month"])
)

for _, row in year_month_combos.iterrows():
    curr_year = row["year"]
    curr_month= row["month"]
    sub_df = all_simulation_results_df.loc[
        (all_simulation_results_df["year"]==curr_year) &
        (all_simulation_results_df["month"]==curr_month)
    ]
    if len(sub_df) == 0:
        continue

    # Observed temperature
    observed_temp = sub_df["temperature"].iloc[0]

    # All simulations for that year-month
    sim_temps = sub_df["simulated_temperature"].values

    # If we have simulations, compute empirical quantiles
    if len(sim_temps) > 0:
        empirical_q = np.quantile(sim_temps, quantiles)
    else:
        empirical_q = np.full_like(quantiles, np.nan)

    # Build result record
    res = {
        "county": county_name,
        "year":   curr_year,
        "month":  curr_month,
        "temperature": observed_temp
    }
    
    # Only compute score for years <= 2024 (matching XGB approach)
    if curr_year <= 2024:
        res["quantile_score"] = quantile_score(observed_temp, empirical_q, quantiles)
    else:
        res["quantile_score"] = np.nan

    results.append(res)

quantile_scores_df = pd.DataFrame(results)

###################### 7) Save Outputs ###############################
quantile_scores_df.to_csv(
    f"/scratch/users/jelazaro/temp_V9_results/sparse_gp/quantile_scores/{county_name}_QS_sparse_GP_lags.csv",
    index=False
)
all_simulation_results_df.to_csv(
    f"/scratch/users/jelazaro/temp_V9_results/sparse_gp/simulation_values/{county_name}_simulation_values_QS_sparse_GP_lags.csv",
    index=False
)

print("GP simulation & quantile score files saved successfully.")
