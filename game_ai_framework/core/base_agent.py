"""
Base Agent Interface
Defines the interface for all AI agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, state_size: int, action_size: int, device: Optional[str] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device if device else 
                                  ("cuda" if torch.cuda.is_available() else "cpu"))
    
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action given state"""
        pass
    
    @abstractmethod
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in memory"""
        pass
    
    @abstractmethod
    def train(self, batch_size: int = 32) -> Optional[float]:
        """Train the agent, return loss if applicable"""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save agent model"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load agent model"""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics"""
        return {}


class TaskSpecificAgent(BaseAgent):
    """Base class for task-specific agents (avoidance, combat, etc.)"""
    
    def __init__(self, state_size: int, action_size: int, task_name: str, 
                 hidden_size: int = 64, device: Optional[str] = None):
        super().__init__(state_size, action_size, device)
        self.task_name = task_name
        self.hidden_size = hidden_size
        self.importance_score = 0.0
        self.usage_count = 0
        self.success_rate = 0.0
    
    @abstractmethod
    def get_confidence(self, state: np.ndarray) -> float:
        """Return confidence score for handling current state (0-1)"""
        pass
    
    def update_importance(self, performance: float) -> None:
        """Update importance score based on performance"""
        alpha = 0.1  # learning rate for importance
        self.importance_score = (1 - alpha) * self.importance_score + alpha * performance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent info to dictionary for logging"""
        return {
            'task_name': self.task_name,
            'importance_score': self.importance_score,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'hidden_size': self.hidden_size
        }


class MetaAgent(BaseAgent):
    """Base class for meta-learning agents that coordinate multiple task agents"""
    
    def __init__(self, state_size: int, action_size: int, device: Optional[str] = None):
        super().__init__(state_size, action_size, device)
        self.task_agents: Dict[str, TaskSpecificAgent] = {}
        self.task_weights: Dict[str, float] = {}
    
    @abstractmethod
    def select_task(self, state: np.ndarray) -> str:
        """Select which task agent to use for current state"""
        pass
    
    @abstractmethod
    def combine_task_outputs(self, task_outputs: Dict[str, np.ndarray]) -> int:
        """Combine outputs from multiple task agents into final action"""
        pass
    
    def add_task_agent(self, task_name: str, agent: TaskSpecificAgent) -> None:
        """Add a task-specific agent"""
        self.task_agents[task_name] = agent
        self.task_weights[task_name] = 1.0 / (len(self.task_agents))
    
    def get_active_tasks(self, state: np.ndarray, threshold: float = 0.1) -> List[str]:
        """Get list of tasks relevant for current state"""
        active = []
        for task_name, agent in self.task_agents.items():
            if agent.get_confidence(state) > threshold:
                active.append(task_name)
        return active