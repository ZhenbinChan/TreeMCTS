"""
TreeMCTS: MCTS-based LLM Rollout Collection

A Python implementation of Monte Carlo Tree Search sampling for collecting
diverse solution trajectories from language models.
"""

__version__ = "0.1.0"
__author__ = "TreeMCTS Contributors"

from .tree_structure import TreeNode, MCTSTree
from .sampling import MCTSSampler

__all__ = [
    "TreeNode",
    "MCTSTree",
    "MCTSSampler",
]
