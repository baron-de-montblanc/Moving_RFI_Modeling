#!/bin/bash

#SBATCH -J pyuvsim
#SBATCH -t 12:00:00
#SBATCH -c 12
#SBATCH --mem=256G
#SBATCH --account=jpober-condo
#SBATCH --array=0-119%20

#SBATCH -e /oscar/data/jpober/jmduchar/Research/Moving_RFI_Modeling/pyuvsim/slurm/_out/arrayjob-%A-%a.out
#SBATCH -o /oscar/data/jpober/jmduchar/Research/Moving_RFI_Modeling/pyuvsim/slurm/_out/arrayjob-%A-%a.out

set -euo pipefail

PARAM_DIR="/oscar/data/jpober/jmduchar/Research/Moving_RFI_Modeling/pyuvsim/setup_files/_obsparams/obspar_test02"

source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pyuvsim

# Each array task processes 10 simulations sequentially.
START_INDEX=$((SLURM_ARRAY_TASK_ID * 10))
END_INDEX=$((START_INDEX + 9))

echo "Starting array task ${SLURM_ARRAY_TASK_ID} on ${HOSTNAME}"
echo "Processing simulations ${START_INDEX} through ${END_INDEX}"

for ((index = START_INDEX; index <= END_INDEX; index++)); do
    t=$(printf "%04d" "${index}")
    PARAM_FILE="${PARAM_DIR}/obsparam${t}.yaml"

    echo "[$(date)] Starting simulation ${index}: ${PARAM_FILE}"

    if [[ ! -f "${PARAM_FILE}" ]]; then
        echo "Parameter file not found: ${PARAM_FILE}" >&2
        exit 1
    fi

    mpirun -n 4 python -c \
        "from pyuvsim.cli import run_param_pyuvsim; run_param_pyuvsim(['${PARAM_FILE}'])"

    echo "[$(date)] Finished simulation ${index}"
done

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID} completed successfully"
