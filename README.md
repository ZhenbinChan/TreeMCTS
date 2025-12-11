# TreeMCTS: MCTS-based LLM Rollout Collection

A Python implementation of MCTS-based tree sampling for collecting LLM rollouts for future reinforcement learning training.

## Overview

TreeMCTS implements a Monte Carlo Tree Search sampling strategy to collect diverse solution trajectories from language models. The system:

1. **Generates** M parallel initial responses to a prompt
2. **Analyzes** each response by splitting into steps and calculating entropy
3. **Branches** from the top-K high-entropy steps
4. **Repeats** the branching process N times to build a complete MCTS tree
5. **Computes** rewards by comparing final answers with ground truth
6. **Exports** the tree structure with rewards for downstream RL training

## Architecture

```
TreeMCTS/
├── src/
│   ├── tree_structure.py      # TreeNode and MCTSTree classes
│   ├── sampling.py            # MCTSSampler for tree construction
│   └── main.py                # Entry point with argument parsing
├── bash_scripts/
│   └── run_sampling.sh        # Launch script with hyperparameter control
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Key Components

#### 1. Tree Structure (`tree_structure.py`)

**TreeNode**: Represents a single step in a solution trajectory
- `node_id`: Unique identifier
- `parent_id`: Parent node reference (for tree structure)
- `step_index`: Position in the response sequence
- `content`: Text content of this step
- `entropy`: Calculated entropy value
- `is_leaf`: Whether this is a terminal node
- `children_ids`: List of child nodes
- `reward`: Computed reward value (updated via backpropagation)

**MCTSTree**: Complete tree for a single sample
- `sample_id`: Dataset sample identifier
- `question`: Original question/prompt
- `prompt`: The used prompt
- `nodes`: Dictionary of all nodes
- `leaf_ids`: Terminal node identifiers
- Methods:
  - `add_node()`: Add a new node to the tree
  - `add_child()`: Add parent-child relationship
  - `backpropagate_reward()`: Compute rewards from leaf correctness
  - `to_dict()`: Serialize to JSON

#### 2. Sampling Module (`sampling.py`)

**MCTSSampler**: Orchestrates the tree building process

Methods:
- `generate_responses()`: Generate M parallel responses using the LLM
- `split_response_into_steps()`: Split response by separator into steps
- `calculate_step_entropy()`: Compute entropy for each step
- `select_high_entropy_steps()`: Select top-K high-entropy steps
- `create_branch_prompt()`: Create continuation prompt from a step
- `build_mcts_tree()`: Main method orchestrating the entire process
- `extract_boxed_answer()`: Extract answer from `\boxed{}` notation

#### 3. Main Entry Point (`main.py`)

Orchestrates:
- Model and tokenizer loading (supports vLLM, DeepSpeed, or standard HF)
- Dataset loading from HuggingFace
- Per-sample tree construction
- Result serialization

## Installation

### Basic Setup

```bash
# Clone or navigate to the project
cd TreeMCTS

# Install dependencies
pip install -r requirements.txt
```

### With GPU Acceleration

#### vLLM (Recommended for inference speed)

```bash
pip install vllm>=0.2.0
```

#### DeepSpeed (For larger models with CPU offload)

```bash
pip install deepspeed>=0.10.0
```

For CPU offload support with CUDA 12.1:
```bash
pip install deepspeed
# May require: pip install flash-attn
```

## Usage

### Command Line Interface

```bash
# Basic usage with default parameters
python src/main.py

# With custom model and parameters
python src/main.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --dataset_id olegbask/AR-LSAT \
    --num_samples 2 \
    --num_branches 2 \
    --top_k_entropy 2 \
    --output_dir ./outputs

# With vLLM acceleration
python src/main.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --use_vllm \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9

# With DeepSpeed and CPU offload
python src/main.py \
    --model_id meta-llama/Llama-2-70b-hf \
    --use_deepspeed \
    --use_cpu_offload
```

### Using Bash Script

```bash
# Make script executable and run
chmod +x bash_scripts/run_sampling.sh
bash bash_scripts/run_sampling.sh

# Or add command-line parameters
bash bash_scripts/run_sampling.sh --use_vllm --max_samples 100
```

### Arguments Reference

**Model & Dataset:**
- `--model_id` (str, default: `meta-llama/Llama-2-7b-hf`): HuggingFace model ID
- `--dataset_id` (str, default: `olegbask/AR-LSAT`): HuggingFace dataset ID
- `--max_samples` (int, optional): Maximum samples to process (for testing)

**Sampling Parameters:**
- `--num_samples` (int, default: 2): Number of parallel initial responses (M)
- `--num_branches` (int, default: 2): Number of branching iterations (N)
- `--top_k_entropy` (int, default: 2): Number of high-entropy steps to branch from (K)
- `--separator` (str, default: `\n\n`): Delimiter for splitting steps

**Output:**
- `--output_dir` (str, default: `./outputs`): Output directory for results
- `--output_format` (str, choices: `json`, `jsonl`, default: `json`): Output format

**Acceleration:**
- `--use_vllm`: Enable vLLM for fast parallel inference
- `--use_deepspeed`: Enable DeepSpeed
- `--use_cpu_offload`: Enable CPU offload for larger models
- `--tensor_parallel_size` (int, default: 1): Tensor parallelism degree
- `--gpu_memory_utilization` (float, default: 0.9): GPU memory ratio for vLLM

**Other:**
- `--seed` (int, default: 42): Random seed
- `--device` (str, default: `cuda`): Device to use (`cuda` or `cpu`)

## Output Format

Results are saved to the output directory as:

### `results.json` or `results.jsonl`

Each sample result contains:

```json
{
  "sample_id": "sample_1",
  "question": "What is...",
  "ground_truth": "A",
  "tree": {
    "sample_id": "sample_1",
    "question": "What is...",
    "prompt": "Question: What is...\nPlease reason...",
    "root_id": "root_abc123",
    "leaf_ids": ["node_def456", "node_ghi789", ...],
    "nodes": {
      "root_abc123": {
        "node_id": "root_abc123",
        "parent_id": null,
        "step_index": -1,
        "content": "Question: What is...",
        "entropy": 0.0,
        "is_leaf": false,
        "children_ids": ["node_jkl012", ...],
        "reward": 0.25,
        "is_correct": null
      },
      "node_def456": {
        "node_id": "node_def456",
        "parent_id": "root_abc123",
        "step_index": 0,
        "content": "First reasoning step...",
        "entropy": 2.15,
        "is_leaf": false,
        "children_ids": ["node_mno345", ...],
        "reward": 0.5,
        "is_correct": null
      },
      ...
    }
  },
  "num_leaves": 8,
  "num_correct_leaves": 2
}
```

### `summary.json`

Aggregated statistics:

```json
{
  "total_samples": 100,
  "successful_samples": 98,
  "avg_num_leaves": 12.5,
  "avg_correct_leaves": 3.2
}
```

## Sampling Process Details

### Step 1: Initial Generation

For each sample, generate M parallel responses to the prompt.

```
Prompt: "Question: What is 2+2?\nPlease reason step by step..."

Response 1: "Step 1: I need to add 2 and 2.\n\nStep 2: 2+2 = 4\n\nAnswer: \boxed{4}"
Response 2: "Step 1: Let me think about this addition.\n\nStep 2: 2 + 2 equals 4.\n\nAnswer: \boxed{4}"
...
```

### Step 2: Entropy Calculation

For each response, split by separator and calculate entropy per step.

Entropy is calculated from token logits:
$$H = -\sum_{i} p_i \log(p_i)$$

where $p_i$ is the token probability.

### Step 3: High-Entropy Step Selection

Select top-K steps with highest entropy values.

```
Response 1 entropies:
  Step 0: 1.2
  Step 1: 2.8  <- High entropy
  Step 2: 0.5

Selected: [Step 1]
```

### Step 4: Branching

For each selected high-entropy step, generate a continuation to create a branch.

```
Branch Point: End of Step 1

New Prompt: "Question: What is 2+2?\nPlease reason step by step...\n\nStep 1: I need to add 2 and 2.\n\n"

Branch Response: "Step 2: Double-checking: 2 + 2 = 4\n\nAnswer: \boxed{4}"
```

### Step 5: Reward Computation

Compute rewards by:

1. Extract answer from `\boxed{}` for each leaf node
2. Compare with ground truth
3. Reward = (Number of correct descendants) / (Total descendants)

## Advanced Configuration

### vLLM Configuration

For optimal performance with vLLM:

```bash
python src/main.py \
    --model_id meta-llama/Llama-2-70b-hf \
    --use_vllm \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.95 \
    --num_samples 4
```

### DeepSpeed Configuration

For large models with CPU offload:

```bash
python src/main.py \
    --model_id meta-llama/Llama-2-70b-hf \
    --use_deepspeed \
    --use_cpu_offload \
    --deepspeed_config ./deepspeed_config.json
```

Example `deepspeed_config.json`:

```json
{
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "optimizer": {
    "type": "Adam",
    "params": {"lr": 1e-5}
  }
}
```

## Performance Tips

1. **Use vLLM** for better inference speed on A100/H100 GPUs
2. **Increase `--num_samples`** for more diverse initial trajectories
3. **Tune `--top_k_entropy`** (typically 2-5) to balance breadth and depth
4. **Use `--tensor_parallel_size`** if model doesn't fit on single GPU
5. **Enable CPU offload** with DeepSpeed for 70B+ models

## Troubleshooting

### OOM (Out of Memory)

```bash
# Use CPU offload with DeepSpeed
python src/main.py --use_deepspeed --use_cpu_offload

# Or reduce batch size and tensor parallelism
python src/main.py --use_vllm --tensor_parallel_size 2 --gpu_memory_utilization 0.8
```

### No GPU Available

```bash
python src/main.py --device cpu
# Note: This will be very slow for large models
```

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# For vLLM (if using --use_vllm)
pip install vllm

# For DeepSpeed (if using --use_deepspeed)
pip install deepspeed
```

## Citation

If you use TreeMCTS in your research, please cite:

```bibtex
@software{treemcts,
  title={TreeMCTS: MCTS-based LLM Rollout Collection},
  author={Your Name},
  year={2024}
}
```

## License

This project is provided as-is for research purposes.

## Contact

For issues or questions, please refer to the GitHub repository.
