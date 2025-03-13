#!/bin/bash
#SBATCH --job-name=model
#SBATCH --output=/home/users/sarteaga/scripts/logs/model.%A_%a.out
#SBATCH --error=/home/users/sarteaga/scripts/logs/model.%A_%a.err
#SBATCH --time=12:00:00 # Adjust as needed
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G # Adjust as needed
#SBATCH --array=1-13 # 13 tasks for the 13 counties

# Set input file path
input_file="/scratch/users/sarteaga/data/CS229/cleaned_weather_ds_v2.csv"
# Define an array of counties
counties=("Fresno" "Kern" "Los Angeles" "Merced" "Orange" "Placer" "Riverside" "Sacramento" "San Bernardino" "San Joaquin" "Solano" "Stanislaus" "Tulare")
# counties=("Los Angeles" "San Bernardino" "San Joaquin")

# Get the county name based on the SLURM_ARRAY_TASK_ID
index=$((SLURM_ARRAY_TASK_ID - 1))
county_name=${counties[$index]}
county_filename=$(echo "${county_name}" | sed 's/ /_/g')
input_dir="/scratch/users/sarteaga/data/CS229/hyperparam/optuna_2000_reduced_reg_auto/"
hyperparam_file="${input_dir}/${county_filename}_best_optuna_trial_results.csv"
output_dir="/scratch/users/sarteaga/data/CS229/modeling_whyperparams/optuna_2000_reduced_reg_auto"

echo -e ${county_filename}

module load python/3.12.1

export PYTHONPATH=/home/users/sarteaga/.local/lib/python3.12/site-packages:$PYTHONPATH
mkdir -p ${output_dir}

# Run the single Python script with the county_name as an argument
python3 /home/users/sarteaga/scripts/model_woptuna_params.py --input_file ${input_file} --hyperparm_file ${hyperparam_file} --county "${county_name}" --output_dir ${output_dir}
