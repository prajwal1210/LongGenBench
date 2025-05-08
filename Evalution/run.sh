#!/bin/bash
#SBATCH --job-name LongGenBench
#SBATCH --gpus-per-node=1
#SBATCH -A m2404_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --time=01:00:00
#SBATCH --qos regular

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export YALIS_CACHE="$SCRATCH/yalis/yalis/external"
export HF_TOKEN=""

module load PrgEnv-gnu/ cudatoolkit/12.4.lua nccl/2.21.5.lua cudnn/9.1.0.lua
module load conda/Miniforge3-24.11.3-0
conda activate yalis-env-latest


export NNODES=$SLURM_JOB_NUM_NODES
export GPUS=$(( NNODES * 1 ))

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=1

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export YALIS_CACHE="$SCRATCH/SpecDec/yalis/yalis/external"

MAX_LENGTH=4000
NUM_GPUS=1
INPUT_DIR="/pscratch/sd/p/prajwal/SpecDec/LongGenBench/Dataset/Dataset_short.json"
OUTPUT_DIR="./results"
export CUDA_VISIBLE_DEVICES=0
CSV_PATH="/pscratch/sd/p/prajwal/SpecDec/LongGenBench/Evalution/results/accuracy_results.csv"

MODEL_TYPE=$1
ATTN_BACKEND=$2
WARMUP=$3
PERCENTILE=$4
MODEL_NAME=$(basename $MODEL_TYPE)

OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${ATTN_BACKEND}_w${WARMUP}_p${PERCENTILE}_maxlen${MAX_LENGTH}.json"

srun -N1 -n $GPUS ./set_env_pm.sh python -u inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE --attn_backend $ATTN_BACKEND --warmup $WARMUP --percentile $PERCENTILE
#python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS
