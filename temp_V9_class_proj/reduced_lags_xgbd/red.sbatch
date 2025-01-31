#!/bin/bash
#SBATCH --job-name=red_lags
#SBATCH --output=red__lags.%A_%a.out
#SBATCH --error=red__lags.%A_%a.err
#SBATCH --time=6:00:00 # Adjust as needed
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G # Adjust as needed
#SBATCH --array=1-13 # 13 tasks for the 13 counties

module load singularity

# Define an array of counties
counties=("Fresno" "Kern" "Los Angeles" "Merced" "Orange" "Placer" "Riverside" "Sacramento" "San Bernardino" "San Joaquin" "Solano" "Stanislaus" "Tulare")

# Get the county name based on the SLURM_ARRAY_TASK_ID
county_name=${counties[$SLURM_ARRAY_TASK_ID-1]}

singularity exec /home/users/jelazaro/python_3.10.13.sif bash -c "
    export PATH=\$HOME/.local/bin:\$PATH
    export PYTHONPATH=\$HOME/.local/lib/python3.10/site-packages:\$PYTHONPATH

    cd /scratch/users/jelazaro/temp_V9_class_proj/reduced_lags_xgbd/

    # Run the single Python script with the county_name as an argument
    python reduced_lags.py \"${county_name}\"
"

