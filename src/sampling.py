"""
MCTS sampling module for generating trajectories and building the tree structure.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from collections import defaultdict
from tree_structure import MCTSTree, TreeNode
import uuid


class MCTSSampler:
    """
    Handles the MCTS sampling process including:
    - Parallel response generation
    - Entropy calculation
    - Tree branch creation
    - Path exploration
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        num_samples: int = 2,
        top_k_entropy: int = 2,
        num_branches: int = 2,
        separator: str = "\n\n",
        device: str = "cuda",
    ):
        """
        Initialize the MCTS sampler.
        
        Args:
            model: The language model (e.g., from vLLM or vllm wrapper)
            tokenizer: The tokenizer
            num_samples: Number of initial responses to generate per sample
            top_k_entropy: Number of high-entropy steps to branch from
            num_branches: Number of times to repeat the branching process
            separator: Separator to split response into steps
            device: Device to use for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.top_k_entropy = top_k_entropy
        self.num_branches = num_branches
        self.separator = separator
        self.device = device
        
        # Ensure tokenizer has a pad token for batch processing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # For decoder-only models, use left-padding to avoid affecting generation
        self.tokenizer.padding_side = "left"
    
    def generate_responses(
        self,
        prompt: str,
        num_responses: int,
        max_length: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple responses in parallel using the model.
        
        Args:
            prompt: The input prompt
            num_responses: Number of responses to generate
            max_length: Maximum length of each response
            
        Returns:
            List of dictionaries containing:
            - 'response': Generated text
            - 'token_ids': Token IDs of generated response
            - 'logits': List of logit vectors for each generated token
            - 'input_ids': Input token IDs
        """
        outputs = []
        
        # Use HuggingFace model with output_scores to get logits
        try:
            # Tokenize with proper padding and truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length - 50,  # Leave room for generation
            ).to(self.device)
            input_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=max_length,
                    num_return_sequences=num_responses,
                    temperature=0.7,
                    top_p=0.9,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            # Extract sequences and scores
            sequences = generation_output.sequences
            scores = generation_output.scores  # List of (batch_size, vocab_size) tensors
            
            # Process each generated sequence
            for i in range(num_responses):
                response_token_ids = sequences[i, input_length:].cpu().tolist()
                response_text = self.tokenizer.decode(
                    response_token_ids,
                    skip_special_tokens=True
                )
                
                # Extract logits for this sequence
                logits_list = []
                for score in scores:
                    # score shape: (num_return_sequences, vocab_size)
                    token_logits = score[i, :].cpu()  # shape: (vocab_size,)
                    logits_list.append(token_logits)
                
                outputs.append({
                    "response": response_text,
                    "token_ids": response_token_ids,
                    "logits": logits_list,  # List of logit vectors
                })
        
        except Exception as e:
            # Fallback: generate without logits tracking
            print(f"Warning: Could not extract logits: {e}")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length - 50,
            ).to(self.device)
            with torch.no_grad():
                outputs_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=max_length,
                    num_return_sequences=num_responses,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            for i in range(num_responses):
                response = self.tokenizer.decode(
                    outputs_ids[i],
                    skip_special_tokens=True
                )
                outputs.append({
                    "response": response,
                    "token_ids": outputs_ids[i].cpu().tolist(),
                    "logits": None,
                })
        
        return outputs
    
    def split_response_into_steps(
        self,
        response: str,
        separator: Optional[str] = None,
    ) -> List[str]:
        """
        Split a response into steps using the separator.
        
        Args:
            response: The full response text
            separator: Separator string (defaults to self.separator)
            
        Returns:
            List of step strings
        """
        if separator is None:
            separator = self.separator
        
        steps = response.split(separator)
        # Filter out empty steps
        steps = [s.strip() for s in steps if s.strip()]
        return steps
    
    def calculate_step_entropy(
        self,
        response_data: Dict[str, Any],
    ) -> Tuple[List[float], float]:
        """
        Calculate entropy for each step in the response based on token logits.
        
        Args:
            response_data: Dictionary with 'response', 'token_ids', and 'logits'
                          'logits' is a list of logit vectors (vocab_size,) for each generated token
            
        Returns:
            Tuple of (step_entropies, mean_entropy) where each step's entropy is the 
            average entropy of its tokens
        """
        response = response_data.get("response", "")
        logits_list = response_data.get("logits", None)
        token_ids = response_data.get("token_ids", [])
        
        steps = self.split_response_into_steps(response)
        step_entropies = []
        
        # If we don't have logits, return zeros
        if logits_list is None or len(logits_list) == 0:
            return [0.0] * len(steps), 0.0
        
        # Tokenize response to map steps to token positions
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        # Build mapping from character positions to token indices
        step_token_ranges = []
        char_pos = 0
        response_text = response
        
        for step in steps:
            # Find where this step starts and ends in the full response
            start_char = response_text.find(step, char_pos)
            if start_char == -1:
                step_token_ranges.append((0, 0))  # Empty range
                continue
            
            end_char = start_char + len(step)
            
            # Find token indices corresponding to this character range
            # This is approximate - we tokenize up to start and end points
            start_tokens = self.tokenizer.encode(response_text[:start_char], add_special_tokens=False)
            end_tokens = self.tokenizer.encode(response_text[:end_char], add_special_tokens=False)
            
            start_idx = len(start_tokens)
            end_idx = len(end_tokens)
            
            step_token_ranges.append((start_idx, end_idx))
            char_pos = end_char
        
        # Calculate entropy for each step
        for start_idx, end_idx in step_token_ranges:
            if start_idx >= end_idx or start_idx >= len(logits_list):
                # No tokens in this step
                step_entropies.append(0.0)
                continue
            
            # Get logits for tokens in this step
            step_logits = logits_list[start_idx:min(end_idx, len(logits_list))]
            
            if len(step_logits) == 0:
                step_entropies.append(0.0)
                continue
            
            # Calculate entropy for each token and average
            token_entropies = []
            for token_logits in step_logits:
                # Convert logits to probabilities using softmax
                if isinstance(token_logits, torch.Tensor):
                    token_logits = token_logits.float()
                else:
                    token_logits = torch.tensor(token_logits, dtype=torch.float32)
                
                # Softmax to get probabilities
                probs = F.softmax(token_logits, dim=-1)
                
                # Calculate entropy: H(p) = -sum(p_i * log(p_i))
                # Avoid log(0) by adding small epsilon
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                token_entropies.append(entropy.item())
            
            # Step entropy is the average of token entropies
            step_entropy = np.mean(token_entropies) if token_entropies else 0.0
            step_entropies.append(step_entropy)
        
        mean_entropy = np.mean(step_entropies) if step_entropies else 0.0
        return step_entropies, mean_entropy
    
    def select_high_entropy_steps(
        self,
        step_entropies: List[float],
        k: Optional[int] = None,
    ) -> List[int]:
        """
        Select indices of top-k high-entropy steps.
        
        Args:
            step_entropies: List of entropy values for each step
            k: Number of steps to select (defaults to self.top_k_entropy)
            
        Returns:
            List of step indices (0-indexed) sorted by entropy
        """
        if k is None:
            k = self.top_k_entropy
        
        k = min(k, len(step_entropies))
        
        # Get indices sorted by entropy (descending)
        top_indices = np.argsort(step_entropies)[-k:][::-1]
        return sorted(top_indices.tolist())
    
    def create_branch_prompt(
        self,
        original_prompt: str,
        response_so_far: str,
        branch_step_index: int,
    ) -> str:
        """
        Create a prompt for branching from a specific step.
        
        Args:
            original_prompt: The original prompt
            response_so_far: The response up to the branching point
            branch_step_index: Index of the step to branch from
            
        Returns:
            New prompt for continuing generation
        """
        steps_so_far = self.split_response_into_steps(response_so_far)
        
        if branch_step_index >= len(steps_so_far):
            # If invalid index, return original prompt
            return original_prompt
        
        # Concatenate all steps up to (but not including) the branch point
        prefix = self.separator.join(steps_so_far[:branch_step_index])
        
        # Return concatenated prompt
        if prefix:
            branch_prompt = original_prompt + self.separator + prefix + self.separator
        else:
            branch_prompt = original_prompt + self.separator
        
        return branch_prompt
    
    def build_mcts_tree(
        self,
        sample_id: str,
        question: str,
        original_prompt: str,
        ground_truth: str,
    ) -> MCTSTree:
        """
        Build the complete MCTS tree for a sample.
        
        Args:
            sample_id: ID of the sample
            question: The original question
            original_prompt: The original prompt
            ground_truth: Ground truth answer
            
        Returns:
            MCTSTree object with complete structure
        """

        tree = MCTSTree(
            sample_id=sample_id,
            question=question,
            prompt=original_prompt,
        )

        
        # Generate initial responses
        initial_responses = self.generate_responses(
            original_prompt,
            num_responses=self.num_samples,
        )


        # Create root node
        root_node = TreeNode(
            node_id=f"root_{uuid.uuid4().hex[:8]}",
            parent_id=None,
            step_index=-1,
            content=original_prompt,
            is_leaf=len(initial_responses) == 0,
        )
        tree.add_node(root_node)

        
        # Process each initial response
        all_leaf_nodes = []
        
        for resp_idx, response_data in enumerate(initial_responses):
            response = response_data["response"]
            steps = self.split_response_into_steps(response)
            
            # Calculate entropy for this response using logits
            step_entropies, _ = self.calculate_step_entropy(response_data)
            
            # Build initial trajectory
            current_parent_id = root_node.node_id
            
            for step_idx, step_content in enumerate(steps):
                node_id = f"node_{uuid.uuid4().hex[:8]}"
                entropy = step_entropies[step_idx] if step_idx < len(step_entropies) else 0.0
                
                node = TreeNode(
                    node_id=node_id,
                    parent_id=current_parent_id,
                    step_index=step_idx,
                    content=step_content,
                    entropy=entropy,
                    is_leaf=True,  # Will update if we branch
                )
                
                tree.add_node(node)
                tree.add_child(current_parent_id, node_id)
                current_parent_id = node_id
            
            all_leaf_nodes.append(current_parent_id)
        
        # Perform branching iterations
        for branch_iter in range(self.num_branches - 1):
            new_leaf_nodes = []
            
            for response_data in initial_responses:
                response = response_data["response"]
                
                # Calculate entropy for this response using logits
                step_entropies, _ = self.calculate_step_entropy(response_data)
                
                # Select high-entropy steps
                high_entropy_indices = self.select_high_entropy_steps(step_entropies)
                
                # For each high-entropy step, create a branch
                for branch_step_idx in high_entropy_indices:
                    # Create branch prompt
                    branch_prompt = self.create_branch_prompt(
                        original_prompt,
                        response,
                        branch_step_idx,
                    )
                    
                    # Generate from this branch point
                    branch_responses = self.generate_responses(
                        branch_prompt,
                        num_responses=1,
                    )
                    
                    if branch_responses:
                        branch_response = branch_responses[0]["response"]
                        branch_steps = self.split_response_into_steps(branch_response)
                        
                        # Extract the actual branch point node
                        steps_so_far = self.split_response_into_steps(response)
                        branch_parent_id = None
                        
                        # Find the node at the branching step
                        current_id = root_node.node_id
                        for step_idx in range(branch_step_idx):
                            found = False
                            for child_id in tree.nodes[current_id].children_ids:
                                child_node = tree.nodes[child_id]
                                if child_node.step_index == step_idx:
                                    current_id = child_id
                                    found = True
                                    break
                            if not found:
                                break
                        
                        branch_parent_id = current_id
                        
                        # Add new nodes for this branch
                        current_parent_id = branch_parent_id
                        
                        for step_idx, step_content in enumerate(branch_steps):
                            node_id = f"node_{uuid.uuid4().hex[:8]}"
                            
                            node = TreeNode(
                                node_id=node_id,
                                parent_id=current_parent_id,
                                step_index=branch_step_idx + step_idx + 1,
                                content=step_content,
                                entropy=0.0,  # Will be updated
                                is_leaf=True,
                            )
                            
                            tree.add_node(node)
                            tree.add_child(current_parent_id, node_id)
                            current_parent_id = node_id
                        
                        new_leaf_nodes.append(current_parent_id)
            
            all_leaf_nodes.extend(new_leaf_nodes)
        
        # Update leaf nodes
        tree.leaf_ids = list(set(all_leaf_nodes))
        for leaf_id in tree.leaf_ids:
            tree.nodes[leaf_id].is_leaf = True
        
        return tree
    
    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from \\boxed{{}} in the response.
        
        Args:
            text: Response text
            
        Returns:
            Extracted answer or None
        """
        match = re.search(r'\\boxed\{(.+?)\}', text)
        if match:
            return match.group(1)
        return None
