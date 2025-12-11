"""
TreeNode and MCTSTree classes for representing the tree structure of MCTS sampling.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json


@dataclass
class TreeNode:
    """
    Represents a single node in the MCTS tree.
    
    Attributes:
        node_id: Unique identifier for the node
        parent_id: ID of the parent node (None for root)
        step_index: Index of the step in the response
        content: The actual text content of this step
        logits: Token logits for this step
        entropy: Average entropy of the step
        is_leaf: Whether this is a leaf node
        children_ids: List of child node IDs
        reward: Reward value for this node (computed during backprop)
        is_correct: Whether the path from root to this node reaches correct answer
    """
    node_id: str
    parent_id: Optional[str]
    step_index: int
    content: str
    logits: Optional[List[float]] = None
    entropy: float = 0.0
    is_leaf: bool = False
    children_ids: List[str] = field(default_factory=list)
    reward: float = 0.0
    is_correct: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "step_index": self.step_index,
            "content": self.content,
            "entropy": self.entropy,
            "is_leaf": self.is_leaf,
            "children_ids": self.children_ids,
            "reward": self.reward,
            "is_correct": self.is_correct,
        }


@dataclass
class MCTSTree:
    """
    Represents the complete MCTS tree for a single sample.
    
    Attributes:
        sample_id: ID of the original sample
        question: The original question
        prompt: The prompt used for generation
        nodes: Dictionary of node_id -> TreeNode
        root_id: ID of the root node
        leaf_ids: List of leaf node IDs
        trajectories: List of trajectories (paths from root to leaves)
    """
    sample_id: str
    question: str
    prompt: str
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    root_id: Optional[str] = None
    leaf_ids: List[str] = field(default_factory=list)
    trajectories: List[List[str]] = field(default_factory=list)  # List of paths (node_id sequences)
    
    def add_node(self, node: TreeNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.node_id] = node
        if node.parent_id is None:
            self.root_id = node.node_id
        if node.is_leaf:
            self.leaf_ids.append(node.node_id)
        else:
            # Remove from leaf list if it's no longer a leaf
            if node.node_id in self.leaf_ids:
                self.leaf_ids.remove(node.node_id)
    
    def update_node(self, node_id: str, **kwargs) -> None:
        """Update a node's attributes."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
    
    def get_node(self, node_id: str) -> TreeNode:
        """Get a node by ID."""
        return self.nodes[node_id]
    
    def get_parent_node(self, node_id: str) -> Optional[TreeNode]:
        """Get the parent node."""
        node = self.nodes.get(node_id)
        if node and node.parent_id:
            return self.nodes.get(node.parent_id)
        return None
    
    def get_ancestors(self, node_id: str) -> List[TreeNode]:
        """Get all ancestor nodes from node to root."""
        ancestors = []
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get the path from a node to the root (node_id sequence)."""
        path = [node_id]
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            path.insert(0, current.parent_id)
            current = self.nodes.get(current.parent_id)
        return path
    
    def get_siblings(self, node_id: str) -> List[TreeNode]:
        """Get all sibling nodes."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return []
        
        parent = self.nodes.get(node.parent_id)
        if not parent:
            return []
        
        siblings = []
        for child_id in parent.children_ids:
            if child_id != node_id:
                siblings.append(self.nodes[child_id])
        return siblings
    
    def add_child(self, parent_id: str, child_id: str) -> None:
        """Add a child-parent relationship."""
        if parent_id in self.nodes:
            parent = self.nodes[parent_id]
            if child_id not in parent.children_ids:
                parent.children_ids.append(child_id)
            # Parent is no longer a leaf
            parent.is_leaf = False
            if parent_id in self.leaf_ids:
                self.leaf_ids.remove(parent_id)
    
    def backpropagate_reward(self, leaf_node_id: str, reward: float) -> None:
        """
        Backpropagate reward from a leaf node to the root.
        """
        path = self.get_path_to_root(leaf_node_id)
        for node_id in path:
            node = self.nodes[node_id]
            node.reward = max(node.reward, reward)  # Or use other aggregation methods
    
    def compute_step_rewards(self, correct_leaf_ids: List[str]) -> None:
        """
        Compute reward for each step based on the fraction of correct paths.
        A step's reward = (number of correct leaf descendants) / (total leaf descendants)
        """
        # Count descendants for each node
        def count_correct_descendants(node_id: str) -> Tuple[int, int]:
            """Count (correct_descendants, total_descendants)."""
            node = self.nodes[node_id]
            
            if node.is_leaf:
                total = 1
                correct = 1 if node_id in correct_leaf_ids else 0
                return correct, total
            
            correct_count = 0
            total_count = 0
            
            for child_id in node.children_ids:
                child_correct, child_total = count_correct_descendants(child_id)
                correct_count += child_correct
                total_count += child_total
            
            return correct_count, total_count
        
        # Compute rewards for all nodes
        def compute_node_reward(node_id: str) -> None:
            correct, total = count_correct_descendants(node_id)
            if total > 0:
                self.nodes[node_id].reward = correct / total
            
            # Recursively compute for children
            node = self.nodes[node_id]
            for child_id in node.children_ids:
                compute_node_reward(child_id)
        
        if self.root_id:
            compute_node_reward(self.root_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire tree to a dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "prompt": self.prompt,
            "root_id": self.root_id,
            "leaf_ids": self.leaf_ids,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "trajectories": self.trajectories,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSTree":
        """Create a tree from a dictionary."""
        tree = cls(
            sample_id=data["sample_id"],
            question=data["question"],
            prompt=data["prompt"],
            root_id=data.get("root_id"),
            leaf_ids=data.get("leaf_ids", []),
            trajectories=data.get("trajectories", []),
        )
        
        for node_id, node_data in data.get("nodes", {}).items():
            node = TreeNode(
                node_id=node_data["node_id"],
                parent_id=node_data["parent_id"],
                step_index=node_data["step_index"],
                content=node_data["content"],
                entropy=node_data.get("entropy", 0.0),
                is_leaf=node_data.get("is_leaf", False),
                children_ids=node_data.get("children_ids", []),
                reward=node_data.get("reward", 0.0),
                is_correct=node_data.get("is_correct", False),
            )
            tree.nodes[node_id] = node
        
        return tree
