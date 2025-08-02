"""
Meta-Learning Agent
Coordinates multiple task-specific agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('..')
from core.base_agent import MetaAgent
from agents.task_agents import create_task_agent


class TaskSelector(nn.Module):
    """Network to select which task agent to use"""
    
    def __init__(self, state_size: int, num_tasks: int):
        super(TaskSelector, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_tasks)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class MetaLearningAgent(MetaAgent):
    """Meta-agent that coordinates task-specific agents"""
    
    def __init__(self, state_size: int, action_size: int, 
                 task_names: List[str] = ['avoidance', 'combat', 'navigation']):
        super().__init__(state_size, action_size)
        
        # Create task-specific agents
        for task in task_names:
            agent = create_task_agent(task, state_size, action_size)
            self.add_task_agent(task, agent)
        
        # Task selector network
        self.task_selector = TaskSelector(state_size, len(task_names)).to(self.device)
        self.selector_optimizer = torch.optim.Adam(self.task_selector.parameters(), lr=0.001)
        
        # Meta-learning parameters
        self.task_history = []
        self.performance_history = {task: [] for task in task_names}
        self.current_task = None
        self.epsilon = 0.2  # For task exploration
        
    def select_task(self, state: np.ndarray) -> str:
        """Select which task agent to use for current state"""
        # Exploration: sometimes randomly select task
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.task_agents.keys()))
        
        # Get task probabilities from selector network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        task_probs = self.task_selector(state_tensor).squeeze().detach().cpu().numpy()
        
        # Get confidence scores from each task agent
        confidences = {}
        for task_name, agent in self.task_agents.items():
            confidences[task_name] = agent.get_confidence(state)
        
        # Combine network output with agent confidences
        task_scores = {}
        task_list = list(self.task_agents.keys())
        for i, task in enumerate(task_list):
            task_scores[task] = task_probs[i] * 0.5 + confidences[task] * 0.5
        
        # Select task with highest combined score
        selected_task = max(task_scores.items(), key=lambda x: x[1])[0]
        self.current_task = selected_task
        
        return selected_task
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using appropriate task agent"""
        # Select task
        task = self.select_task(state)
        
        # Use selected task agent to choose action
        action = self.task_agents[task].act(state, training)
        
        # Record task usage
        self.task_history.append(task)
        
        return action
    
    def combine_task_outputs(self, task_outputs: Dict[str, np.ndarray]) -> int:
        """Combine outputs from multiple task agents (alternative approach)"""
        # This method allows using multiple agents simultaneously
        # For now, we use single task selection, but this could be extended
        
        # Weight outputs by task importance
        weighted_sum = np.zeros(self.action_size)
        total_weight = 0
        
        for task, output in task_outputs.items():
            weight = self.task_weights.get(task, 1.0)
            weighted_sum += output * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum /= total_weight
        
        return np.argmax(weighted_sum)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in appropriate task agent's memory"""
        if self.current_task and self.current_task in self.task_agents:
            # Store in task-specific agent
            self.task_agents[self.current_task].remember(state, action, reward, 
                                                        next_state, done)
            
            # Update task performance
            self.performance_history[self.current_task].append(reward)
            self.task_agents[self.current_task].update_importance(reward)
    
    def train(self, batch_size: int = 32) -> Optional[float]:
        """Train all components"""
        total_loss = 0
        num_trained = 0
        
        # Train each task agent
        for task_name, agent in self.task_agents.items():
            loss = agent.train(batch_size)
            if loss is not None:
                total_loss += loss
                num_trained += 1
        
        # Train task selector
        if len(self.task_history) >= batch_size:
            selector_loss = self._train_task_selector(batch_size)
            if selector_loss is not None:
                total_loss += selector_loss
                num_trained += 1
        
        # Update task weights based on performance
        self._update_task_weights()
        
        # Decay exploration
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return total_loss / num_trained if num_trained > 0 else None
    
    def _train_task_selector(self, batch_size: int) -> Optional[float]:
        """Train the task selector network"""
        # This is a simplified version - in practice, you'd want to train
        # the selector based on which tasks led to better outcomes
        
        # For now, we'll train it to match the confidence scores
        if not hasattr(self, 'selector_memory'):
            self.selector_memory = []
        
        # Collect recent task selections and their outcomes
        # This would need to be implemented based on tracking state-task-reward triplets
        
        return None  # Placeholder
    
    def _update_task_weights(self) -> None:
        """Update task weights based on recent performance"""
        window_size = 100
        
        for task in self.task_agents:
            if len(self.performance_history[task]) >= window_size:
                recent_performance = np.mean(self.performance_history[task][-window_size:])
                # Update weight based on performance
                self.task_weights[task] = 0.5 + 0.5 * np.tanh(recent_performance / 10)
    
    def save(self, filepath: str) -> None:
        """Save all components"""
        checkpoint = {
            'task_selector': self.task_selector.state_dict(),
            'task_weights': self.task_weights,
            'task_history': self.task_history[-1000:],  # Save recent history
            'performance_history': {k: v[-1000:] for k, v in self.performance_history.items()},
            'epsilon': self.epsilon
        }
        
        # Save meta-agent state
        torch.save(checkpoint, filepath)
        
        # Save each task agent
        for task_name, agent in self.task_agents.items():
            task_filepath = filepath.replace('.pth', f'_{task_name}.pth')
            agent.save(task_filepath)
    
    def load(self, filepath: str) -> None:
        """Load all components"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.task_selector.load_state_dict(checkpoint['task_selector'])
        self.task_weights = checkpoint['task_weights']
        self.task_history = checkpoint.get('task_history', [])
        self.performance_history = checkpoint.get('performance_history', 
                                                {task: [] for task in self.task_agents})
        self.epsilon = checkpoint.get('epsilon', 0.01)
        
        # Load each task agent
        for task_name, agent in self.task_agents.items():
            task_filepath = filepath.replace('.pth', f'_{task_name}.pth')
            try:
                agent.load(task_filepath)
            except FileNotFoundError:
                print(f"Warning: Could not load {task_name} agent from {task_filepath}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics"""
        metrics = {
            'epsilon': self.epsilon,
            'num_tasks': len(self.task_agents)
        }
        
        # Task usage statistics
        if self.task_history:
            recent_history = self.task_history[-1000:]
            for task in self.task_agents:
                usage_rate = recent_history.count(task) / len(recent_history)
                metrics[f'{task}_usage_rate'] = usage_rate
        
        # Task performance
        for task, history in self.performance_history.items():
            if history:
                metrics[f'{task}_avg_reward'] = np.mean(history[-100:])
        
        # Individual agent metrics
        for task_name, agent in self.task_agents.items():
            metrics[f'{task_name}_importance'] = agent.importance_score
            metrics[f'{task_name}_usage_count'] = agent.usage_count
        
        return metrics
    
    def analyze_transfer_potential(self, new_game_state: np.ndarray) -> Dict[str, float]:
        """Analyze how well current skills might transfer to a new game"""
        transfer_scores = {}
        
        for task_name, agent in self.task_agents.items():
            # Get confidence score for new game state
            confidence = agent.get_confidence(new_game_state)
            
            # Factor in agent's historical performance
            if self.performance_history[task_name]:
                avg_performance = np.mean(self.performance_history[task_name][-100:])
                transfer_score = confidence * (1 + np.tanh(avg_performance / 10))
            else:
                transfer_score = confidence * 0.5
            
            transfer_scores[task_name] = transfer_score
        
        return transfer_scores