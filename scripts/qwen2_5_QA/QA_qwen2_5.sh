#!/bin/bash

#SBATCH --job-name=QA_Evaluation
#SBATCH --mail-user=josuetf@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/josuetf/slurm/%x_%j.out
#SBATCH --partition=spgpu
#SBATCH --time=14-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --account=chaijy2
#SBATCH --chdir=/nfs/turbo/coe-chaijy-unreplicated/josuetf
#SBATCH --export=DISABLE_VERSION_CHECK=1
#SBATCH --exclude=gl1527

# run job
module load cuda/11.8.0
module load python3.11-anaconda/2024.02
source activate base
conda activate vllm

# Debug: Check GPU availability
echo "=== GPU Configuration ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPU(s)')"
echo "========================="

# Explicitly set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

python qwen_vl_safety_eval_vllm.py  --model /nfs/turbo/coe-chaijy-unreplicated/josuetf/models/qwen-2_5_vl-7b-instruct-local  --output qwen_2_5_7b_qa_results_vllm_4bit_interleaved.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit
python qwen_vl_safety_eval_vllm.py  --model /nfs/turbo/coe-chaijy-unreplicated/josuetf/models/qwen-2_5_vl-7b-instruct-local  --output qwen_2_5_7b_qa_results_vllm_4bit_no_metadata_interleaved.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --no-metadata
python qwen_vl_safety_eval_vllm.py  --model /nfs/turbo/coe-chaijy-unreplicated/josuetf/models/qwen-2_5_vl-32b-instruct-local  --output qwen_2_5_32b_qa_results_vllm_4bit_interleaved.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit
python qwen_vl_safety_eval_vllm.py  --model /nfs/turbo/coe-chaijy-unreplicated/josuetf/models/qwen-2_5_vl-32b-instruct-local  --output qwen_2_5_32b_qa_results_vllm_4bit_no_metadata_interleaved.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --no-metadata
python qwen_vl_safety_eval_vllm.py  --model /nfs/turbo/coe-chaijy-unreplicated/josuetf/models/qwen-2_5_vl-72b-instruct-local  --output qwen_2_5_72b_qa_results_vllm_4bit_interleaved.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit
python qwen_vl_safety_eval_vllm.py  --model /nfs/turbo/coe-chaijy-unreplicated/josuetf/models/qwen-2_5_vl-72b-instruct-local  --output qwen_2_5_72b_qa_results_vllm_4bit_no_metadata_interleaved.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --no-metadata