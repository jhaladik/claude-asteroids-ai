"""
Base Game Interface
Defines the universal interface that all games must implement
for compatibility with the meta-learning framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class GameState:
    """Universal game state representation"""
    # Core state information
    frame: np.ndarray  # Visual representation
    entities: List[Dict[str, Any]]  # List of game entities
    
    # Semantic features
    player_position: Tuple[float, float]
    threats: List[Dict[str, Any]]  # Positions and velocities of threats
    collectibles: List[Dict[str, Any]]  # Positions of collectibles
    
    # Game-specific metadata
    score: int
    lives: int
    level: int
    time_remaining: Optional[float]
    
    # Task-relevant features
    distances_to_threats: List[float]
    distances_to_collectibles: List[float]
    safe_zones: List[Tuple[float, float]]
    
    def to_vector(self) -> np.ndarray:
        """Convert state to fixed-size vector for neural networks"""
        # This will be implemented based on specific needs
        pass


@dataclass
class TaskContext:
    """Information about which tasks are relevant in current state"""
    avoidance_priority: float  # 0-1, how important is avoiding threats
    collection_priority: float  # 0-1, how important is collecting items
    combat_priority: float  # 0-1, how important is attacking
    navigation_priority: float  # 0-1, how important is pathfinding
    survival_priority: float  # 0-1, how important is staying alive
    
    def get_active_tasks(self, threshold: float = 0.1) -> List[str]:
        """Return list of active tasks above threshold"""
        tasks = []
        if self.avoidance_priority > threshold:
            tasks.append('avoidance')
        if self.collection_priority > threshold:
            tasks.append('collection')
        if self.combat_priority > threshold:
            tasks.append('combat')
        if self.navigation_priority > threshold:
            tasks.append('navigation')
        if self.survival_priority > threshold:
            tasks.append('survival')
        return tasks


class BaseGame(ABC):
    """Abstract base class for all games"""
    
    def __init__(self, width: int = 800, height: int = 600, render: bool = True):
        self.width = width
        self.height = height
        self.render_enabled = render
        self.reset()
    
    @abstractmethod
    def reset(self) -> GameState:
        """Reset game to initial state"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Execute one game step
        Returns: (next_state, reward, done, info)
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render the game if render is enabled"""
        pass
    
    @abstractmethod
    def get_state(self) -> GameState:
        """Get current game state in universal format"""
        pass
    
    @abstractmethod
    def get_action_space(self) -> int:
        """Return number of possible actions"""
        pass
    
    @abstractmethod
    def get_action_meanings(self) -> List[str]:
        """Return human-readable action descriptions"""
        pass
    
    @abstractmethod
    def analyze_task_context(self) -> TaskContext:
        """Analyze current state to determine task priorities"""
        pass
    
    @abstractmethod
    def get_state_vector(self) -> np.ndarray:
        """Get state as fixed-size vector for neural networks"""
        pass
    
    @abstractmethod
    def get_semantic_features(self) -> Dict[str, float]:
        """
        Get high-level semantic features that transfer across games
        e.g., threat_level, collection_opportunity, escape_routes
        """
        pass
    
    def close(self) -> None:
        """Clean up resources"""
        pass