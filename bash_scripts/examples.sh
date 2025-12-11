#!/bin/bash

# Example usage scripts for TreeMCTS

echo "TreeMCTS Example Usage Scripts"
echo "=============================="
echo ""

# Example 1: Basic usage with small test set
echo "Example 1: Basic usage with 10 samples"
echo "Command: python src/main.py --max_samples 10"
echo ""

# Example 2: With vLLM acceleration
echo "Example 2: With vLLM acceleration"
echo "Command: python src/main.py --use_vllm --tensor_parallel_size 1 --max_samples 10"
echo ""

# Example 3: With different hyperparameters
echo "Example 3: More diverse sampling"
echo "Command: python src/main.py \\"
echo "    --num_samples 4 \\"
echo "    --num_branches 3 \\"
echo "    --top_k_entropy 3 \\"
echo "    --max_samples 10"
echo ""

# Example 4: With custom model
echo "Example 4: Using a different model"
echo "Command: python src/main.py \\"
echo "    --model_id mistralai/Mistral-7B-v0.1 \\"
echo "    --max_samples 5"
echo ""

# Example 5: With DeepSpeed for large model
echo "Example 5: Large model with DeepSpeed"
echo "Command: python src/main.py \\"
echo "    --model_id meta-llama/Llama-2-70b-hf \\"
echo "    --use_deepspeed \\"
echo "    --use_cpu_offload \\"
echo "    --max_samples 5"
echo ""

# Example 6: Full production run
echo "Example 6: Full production run"
echo "Command: bash bash_scripts/run_sampling.sh \\"
echo "    --model_id meta-llama/Llama-2-7b-hf \\"
echo "    --use_vllm \\"
echo "    --num_samples 4 \\"
echo "    --num_branches 3 \\"
echo "    --top_k_entropy 3 \\"
echo "    --output_dir ./results"
echo ""

echo "=============================="
echo ""
echo "To run any of these examples, navigate to the project root directory"
echo "and execute the command."
echo ""
echo "For more details, see README.md"
