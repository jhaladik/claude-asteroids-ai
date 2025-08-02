"""
Task-Specific Agents
Specialized agents for different game tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Optional, Dict, Any
import sys
sys.path.append('..')
from core.base_agent import TaskSpecificAgent


class TaskNetwork(nn.Module):
    """Neural network for task-specific agents"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(TaskNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class AvoidanceAgent(TaskSpecificAgent):
    """Agent specialized in avoiding threats"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__(state_size, action_size, "avoidance", hidden_size)
        
        self.network = TaskNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)
        self.epsilon = 0.1  # Lower epsilon for specialized agents
        self.gamma = 0.95
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action focused on avoiding threats"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.network(state_tensor)
        
        # Bias towards movement actions when threats are detected
        if self._detect_nearby_threat(state):
            # Reduce probability of "do nothing" action
            q_values[0][0] -= 2.0
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def _detect_nearby_threat(self, state: np.ndarray) -> bool:
        """Simple threat detection from state vector"""
        # Assuming threat distances are in specific indices
        # This would be customized based on state representation
        threat_threshold = 0.3
        # Check first threat distance (normalized)
        if len(state) > 20:  # Ensure state has threat info
            return state[20] < threat_threshold  # First threat distance
        return False
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Return confidence for handling avoidance in current state"""
        # High confidence when threats are nearby
        if self._detect_nearby_threat(state):
            return 0.9
        # Low confidence when no threats
        return 0.1
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience with avoidance-specific reward shaping"""
        # Enhance rewards for successful avoidance
        if not done and reward >= 0:
            reward += 0.5  # Bonus for staying alive
        
        self.memory.append((state, action, reward, next_state, done))
        self.usage_count += 1
    
    def train(self, batch_size: int = 32) -> Optional[float]:
        """Train the avoidance network"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device)
        
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath: str) -> None:
        """Save agent model"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'task_name': self.task_name,
            'importance_score': self.importance_score,
            'usage_count': self.usage_count
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.importance_score = checkpoint.get('importance_score', 0)
        self.usage_count = checkpoint.get('usage_count', 0)


class CombatAgent(TaskSpecificAgent):
    """Agent specialized in combat/shooting tasks"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__(state_size, action_size, "combat", hidden_size)
        
        self.network = TaskNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)
        self.epsilon = 0.1
        self.gamma = 0.95
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action focused on combat"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.network(state_tensor)
        
        # Bias towards shooting when targets are aligned
        if self._target_aligned(state):
            # Assuming action 4 is shoot
            if self.action_size > 4:
                q_values[0][4] += 2.0
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def _target_aligned(self, state: np.ndarray) -> bool:
        """Check if a target is aligned for shooting"""
        # This would check angle to nearest threat
        # Simplified version
        return random.random() < 0.3  # 30% of the time
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Return confidence for handling combat in current state"""
        # High confidence when targets are at medium range
        if len(state) > 20:
            threat_dist = state[20]  # First threat distance
            if 0.2 < threat_dist < 0.5:  # Medium range
                return 0.8
        return 0.2
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience with combat-specific reward shaping"""
        # Enhance rewards for successful hits
        # This would need game-specific info about whether we hit something
        self.memory.append((state, action, reward, next_state, done))
        self.usage_count += 1
    
    def train(self, batch_size: int = 32) -> Optional[float]:
        """Train the combat network"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device)
        
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath: str) -> None:
        """Save agent model"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'task_name': self.task_name,
            'importance_score': self.importance_score,
            'usage_count': self.usage_count
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.importance_score = checkpoint.get('importance_score', 0)
        self.usage_count = checkpoint.get('usage_count', 0)


class NavigationAgent(TaskSpecificAgent):
    """Agent specialized in navigation and positioning"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__(state_size, action_size, "navigation", hidden_size)
        
        self.network = TaskNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)
        self.epsilon = 0.1
        self.gamma = 0.95
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action focused on navigation"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.network(state_tensor)
        
        # Bias towards movement when far from center
        if self._far_from_center(state):
            # Encourage thrust action
            if self.action_size > 3:
                q_values[0][3] += 1.0
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def _far_from_center(self, state: np.ndarray) -> bool:
        """Check if player is far from center"""
        # Assuming player position is in first two elements (normalized)
        if len(state) >= 2:
            center_dist = np.sqrt((state[0] - 0.5)**2 + (state[1] - 0.5)**2)
            return center_dist > 0.3
        return False
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Return confidence for handling navigation in current state"""
        # High confidence when we need to reposition
        if self._far_from_center(state):
            return 0.7
        return 0.3
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
        self.usage_count += 1
    
    def train(self, batch_size: int = 32) -> Optional[float]:
        """Train the navigation network"""
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device)
        
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath: str) -> None:
        """Save agent model"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'task_name': self.task_name,
            'importance_score': self.importance_score,
            'usage_count': self.usage_count
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.importance_score = checkpoint.get('importance_score', 0)
        self.usage_count = checkpoint.get('usage_count', 0)


# Factory function to create task agents
def create_task_agent(task_name: str, state_size: int, action_size: int, 
                     hidden_size: int = 64) -> TaskSpecificAgent:
    """Create a task-specific agent by name"""
    task_agents = {
        'avoidance': AvoidanceAgent,
        'combat': CombatAgent,
        'navigation': NavigationAgent
    }
    
    if task_name not in task_agents:
        raise ValueError(f"Unknown task: {task_name}")
    
    return task_agents[task_name](state_size, action_size, hidden_size)