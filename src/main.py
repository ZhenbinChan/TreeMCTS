"""
Main entry point for TreeMCTS sampling system.

Usage:
    python main.py \
        --model_id meta-llama/Llama-2-7b-hf \
        --dataset_id olegbask/AR-LSAT \
        --num_samples 2 \
        --num_branches 2 \
        --top_k_entropy 2 \
        --output_dir ./outputs
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TreeMCTS sampling for MCTS-based LLM rollout collection")
    
    # Model and dataset configuration
    parser.add_argument("--model_id",type=str,default="meta-llama/Llama-2-7b-hf",help="HuggingFace model ID")
    parser.add_argument("--dataset_id",type=str,default="olegbask/AR-LSAT",help="HuggingFace dataset ID")
    
    # Sampling parameters
    parser.add_argument("--num_samples",type=int,default=2,help="Number of parallel initial responses (M)")
    parser.add_argument("--num_branches",type=int,default=2,help="Number of branching iterations (N)")
    parser.add_argument("--top_k_entropy",type=int,default=2,help="Number of high-entropy steps to branch from (K)")
    parser.add_argument("--separator",type=str,default="\n\n",help="Separator for splitting response into steps")
    
    # Output configuration
    parser.add_argument("--output_dir",type=str,default="./outputs",help="Output directory for results")
    parser.add_argument("--output_format",type=str,default="json",choices=["json", "jsonl"],help="Output format")
    
    # DeepSpeed configuration
    parser.add_argument("--use_deepspeed",action="store_true",help="Use DeepSpeed for inference")
    
    parser.add_argument("--deepspeed_config",type=str,default=None,help="Path to DeepSpeed config file")
    parser.add_argument("--use_cpu_offload",action="store_true",help="Enable CPU offload for larger models")
    
    # vLLM configuration
    parser.add_argument("--use_vllm",action="store_true",help="Use vLLM for inference")
    parser.add_argument("--tensor_parallel_size",type=int,default=1,help="Tensor parallelism size")
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.9,help="GPU memory utilization ratio for vLLM")
    
    # Processing configuration
    parser.add_argument("--max_samples",type=int,default=None,help="Maximum number of samples to process (for testing)")
    parser.add_argument("--seed",type=int,default=42,help="Random seed")
    parser.add_argument("--device",type=str,default="cuda",help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    return str(output_path)


def load_model_and_tokenizer(model_id: str, args: argparse.Namespace):
    """
    Load model and tokenizer.
    
    Args:
        model_id: HuggingFace model ID
        args: Argument namespace
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_id}")
    
    if args.use_vllm:
        from vllm import LLM, SamplingParams
        model = LLM(
            model=model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tokenizer = model.get_tokenizer()
    elif args.use_deepspeed:
        import deepspeed
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Initialize DeepSpeed
        if args.deepspeed_config:
            with open(args.deepspeed_config) as f:
                ds_config = json.load(f)
        else:
            # Default config
            ds_config = {
                "train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {"lr": 1e-5}
                },
            }
            if args.use_cpu_offload:
                ds_config["zero_optimization"] = {
                    "stage": 3,
                    "offload_optimizer": {"device": "cpu"},
                    "offload_param": {"device": "cpu"}
                }
        
        model = deepspeed.initialize(model=model, config_dict=ds_config)[0]
    else:
        # Standard HuggingFace loading
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if args.device == "cuda" else "cpu",
        )
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def load_dataset(dataset_id: str, max_samples: Optional[int] = None):
    """
    Load dataset from HuggingFace.
    
    Args:
        dataset_id: HuggingFace dataset ID
        max_samples: Maximum number of samples to load
        
    Yields:
        Dataset samples
    """

    from datasets import load_dataset
    
    logger.info(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id)
    
    # Assuming 'train' split exists
    split = dataset['train'] if 'train' in dataset else list(dataset.keys())[0]
    
    if max_samples:
        split = split.select(range(min(max_samples, len(split))))
    
    logger.info(f"Loaded {len(split)} samples")
    
    for sample in split:
        yield sample
            



def process_sample(
    sample: Dict[str, Any],
    sampler,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample and build its MCTS tree.
    
    Args:
        sample: Dataset sample
        sampler: MCTSSampler instance
        args: Argument namespace
        
    Returns:
        Dictionary with sample ID and tree data, or None if processing failed
    """

    # Extract relevant fields from sample
    # These field names depend on the specific dataset structure
    # For AR-LSAT, common fields might be 'question', 'answer', etc.
    sample_id = sample.get('id', sample.get('sample_id', str(hash(str(sample)))))
    context = sample.get('context', '')
    option_list = sample.get('answers', [])
    options = ""
    for i in range(len(option_list)):
        options += f"\n\n({chr(65 + i)}) {option_list[i]}"
    question = sample.get('question', sample.get('text', ''))
    ground_truth = sample.get('answer', sample.get('label', ''))
    
    if not question or not ground_truth:
        logger.warning(f"Sample {sample_id} missing required fields")
        return None
    
    # Create prompt - adapt based on your requirements
    prompt = (
        f"Question: {context}\n\n{question}\n\n"
        f"Options: {options}\n\n"
        f"Please reason step by step with steps separated by \"\\n\\n\" and put the index of the correct answer within \\boxed{{}}."
    )
    
    # Build MCTS tree
    tree = sampler.build_mcts_tree(
        sample_id=sample_id,
        question=question,
        original_prompt=prompt,
        ground_truth=ground_truth,
    )
    
    # Extract answers from leaf nodes and compute rewards
    correct_leaf_ids = []
    for leaf_id in tree.leaf_ids:
        leaf_node = tree.nodes[leaf_id]
        extracted_answer = sampler.extract_boxed_answer(leaf_node.content)
        
        # Compare with ground truth
        if extracted_answer and extracted_answer.strip() == str(ground_truth).strip():
            correct_leaf_ids.append(leaf_id)
    
    # Compute rewards for all nodes based on correct leaf descendants
    tree.compute_step_rewards(correct_leaf_ids)
    
    return {
        'sample_id': sample_id,
        'question': question,
        'ground_truth': ground_truth,
        'tree': tree.to_dict(),
        'num_leaves': len(tree.leaf_ids),
        'num_correct_leaves': len(correct_leaf_ids),
    }

def save_results(
    results: list,
    output_dir: str,
    output_format: str = "json"
):
    """
    Save results to file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
        output_format: Output format (json or jsonl)
    """
    output_path = Path(output_dir)
    
    if output_format == "json":
        output_file = output_path / "results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_file}")
        
    elif output_format == "jsonl":
        output_file = output_path / "results.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        logger.info(f"Saved results to {output_file}")
    
    # Also save summary statistics
    summary = {
        'total_samples': len(results),
        'successful_samples': len([r for r in results if r is not None]),
        'avg_num_leaves': np.mean([r['num_leaves'] for r in results if r]),
        'avg_correct_leaves': np.mean([r['num_correct_leaves'] for r in results if r]),
    }
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args)
    
    # Initialize sampler
    from sampling import MCTSSampler
    sampler = MCTSSampler(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        top_k_entropy=args.top_k_entropy,
        num_branches=args.num_branches,
        separator=args.separator,
        device=args.device,
    )
    
    # Process dataset
    results = []
    dataset_loader = load_dataset(args.dataset_id, max_samples=args.max_samples)
    
    for sample in tqdm(dataset_loader, desc="Processing samples"):
        result = process_sample(sample, sampler, args)
        if result:
            results.append(result)
    
    # Save results
    save_results(results, output_dir, args.output_format)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
