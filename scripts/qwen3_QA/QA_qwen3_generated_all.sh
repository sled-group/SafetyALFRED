#!/bin/bash

#SBATCH --job-name=QA_Qwen3_Generated_All
#SBATCH --mail-user=josuetf@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/josuetf/slurm/%x_%j.out
#SBATCH --partition=spgpu
#SBATCH --time=5-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --account=chaijy0
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

echo "=========================================="
echo "Starting Qwen3-VL Generated Trajectories QA Evaluation"
echo "Processing: Normal, Complex, and Complex Zeroshot modes"
echo "=========================================="

# Normal mode (no complex)
echo ""
echo "=== Running Normal Mode (No Complex) ==="
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-4B-Instruct --output qwen3_vl_4b_qa_results_vllm_4bit_interleaved_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-4B-Instruct --output qwen3_vl_4b_qa_results_vllm_4bit_no_metadata_interleaved_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --no-metadata --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-8B-Instruct --output qwen3_vl_8b_qa_results_vllm_4bit_interleaved_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-8B-Instruct --output qwen3_vl_8b_qa_results_vllm_4bit_no_metadata_interleaved_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --no-metadata --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-32B-Instruct --output qwen3_vl_32b_qa_results_vllm_4bit_interleaved_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-32B-Instruct --output qwen3_vl_32b_qa_results_vllm_4bit_no_metadata_interleaved_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 64 --quantization bitsandbytes --load-in-4bit --no-metadata --generated-only

# Complex mode (with few-shot examples)
echo ""
echo "=== Running Complex Mode (Few-shot) ==="
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-4B-Instruct --output qwen3_vl_4b_qa_results_vllm_4bit_interleaved_complex_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --complex --max-model-len 50000 --super-batch-per-category --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-4B-Instruct --output qwen3_vl_4b_qa_results_vllm_4bit_no_metadata_interleaved_complex_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --no-metadata --complex --max-model-len 50000 --super-batch-per-category --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-8B-Instruct --output qwen3_vl_8b_qa_results_vllm_4bit_interleaved_complex_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --complex --max-model-len 50000 --super-batch-per-category --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-8B-Instruct --output qwen3_vl_8b_qa_results_vllm_4bit_no_metadata_interleaved_complex_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --no-metadata --complex --max-model-len 50000 --super-batch-per-category --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-32B-Instruct --output qwen3_vl_32b_qa_results_vllm_4bit_interleaved_complex_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --complex --max-model-len 50000 --super-batch-per-category --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-32B-Instruct --output qwen3_vl_32b_qa_results_vllm_4bit_no_metadata_interleaved_complex_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --no-metadata --complex --max-model-len 50000 --super-batch-per-category --generated-only

# Complex Zeroshot mode (no examples)
echo ""
echo "=== Running Complex Zeroshot Mode (No Examples) ==="
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-4B-Instruct --output qwen3_vl_4b_qa_results_vllm_4bit_interleaved_complex_zeroshot_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --complex --max-model-len 50000 --super-batch-per-category --no-examples --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-4B-Instruct --output qwen3_vl_4b_qa_results_vllm_4bit_no_metadata_interleaved_complex_zeroshot_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --no-metadata --complex --max-model-len 50000 --super-batch-per-category --no-examples --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-8B-Instruct --output qwen3_vl_8b_qa_results_vllm_4bit_interleaved_complex_zeroshot_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --complex --max-model-len 50000 --super-batch-per-category --no-examples --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-8B-Instruct --output qwen3_vl_8b_qa_results_vllm_4bit_no_metadata_interleaved_complex_zeroshot_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --no-metadata --complex --max-model-len 50000 --super-batch-per-category --no-examples --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-32B-Instruct --output qwen3_vl_32b_qa_results_vllm_4bit_interleaved_complex_zeroshot_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --complex --max-model-len 50000 --super-batch-per-category --no-examples --generated-only
python qwen_vl_safety_eval_vllm.py --model Qwen/Qwen3-VL-32B-Instruct --output qwen3_vl_32b_qa_results_vllm_4bit_no_metadata_interleaved_complex_zeroshot_generated.jsonl --tensor-parallel-size 2 --max-num-seqs 32 --quantization bitsandbytes --load-in-4bit --no-metadata --complex --max-model-len 50000 --super-batch-per-category --no-examples --generated-only

echo ""
echo "=========================================="
echo "All Qwen3-VL Generated Trajectories QA Evaluation Complete!"
echo "=========================================="
