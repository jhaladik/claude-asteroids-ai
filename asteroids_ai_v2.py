import pygame
import random
import math
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import csv
import datetime
import statistics
from typing import Dict, List, Tuple, Optional

# Load AI parameters configuration
def load_ai_parameters(param_set="default"):
    """Load AI parameters from configuration file"""
    try:
        with open("ai_parameters.json", 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        print("ai_parameters.json not found, using default parameters")
        return None

# Initialize Pygame first
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Advanced Asteroids - AI Driven")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Game parameters
player_size = 20
player_speed = 3
bullet_speed = 8
asteroid_speed = 1
asteroid_spawn_time = 120

# Initialize game state variables
player_pos = [width // 2, height // 2]
player_angle = 0
bullets = []
asteroids = []
explosions = []
level = 1
score = 0
asteroids_destroyed = 0
game_over = False
frame_count = 0

# Font
font = pygame.font.Font(None, 36)

# Highscores
highscores = []
highscore_file = "highscores.json"

# Stars for background
stars = [(random.randint(0, width), random.randint(0, height)) for _ in range(100)]

# Helper functions (moved before AI code)
def check_collision(pos1, pos2, size):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < size + player_size // 2

def spawn_asteroid():
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return [random.randint(0, width), 0, random.randint(20, 50), random.choice([RED, BLUE, YELLOW]), 0]
    elif side == "bottom":
        return [random.randint(0, width), height, random.randint(20, 50), random.choice([RED, BLUE, YELLOW]), 0]
    elif side == "left":
        return [0, random.randint(0, height), random.randint(20, 50), random.choice([RED, BLUE, YELLOW]), 0]
    else:
        return [width, random.randint(0, height), random.randint(20, 50), random.choice([RED, BLUE, YELLOW]), 0]

# Comprehensive Training Logger for AI Learning Analysis
class TrainingLogger:
    """Tracks all training metrics for AI learning curve analysis"""
    
    def __init__(self, agent_type: str, log_dir: str = "training_logs"):
        self.agent_type = agent_type
        self.log_dir = log_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metric containers
        self.episode_metrics = []
        self.training_steps = []
        self.parameter_history = []
        self.task_metrics = []  # For meta-learning
        
        # Current episode tracking
        self.current_episode = {
            'episode': 0,
            'score': 0,
            'steps': 0,
            'epsilon': 0,
            'avg_q_value': 0,
            'avg_loss': 0,
            'losses': [],
            'q_values': [],
            'actions': [],
            'rewards': [],
            'survival_time': 0,
            'asteroids_destroyed': 0,
            'accuracy': 0,
            'exploration_rate': 0
        }
        
        # Training parameters
        self.parameters = {}
        
        # Create CSV writers
        self._init_csv_files()
        
    def _init_csv_files(self):
        """Initialize CSV files for different metrics"""
        # Episode summary file
        self.episode_file = os.path.join(self.log_dir, f"{self.agent_type}_episodes_{self.timestamp}.csv")
        with open(self.episode_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'score', 'steps', 'epsilon', 'avg_q_value', 'avg_loss',
                           'survival_time', 'asteroids_destroyed', 'accuracy', 'exploration_rate',
                           'reward_mean', 'reward_std', 'action_entropy'])
        
        # Detailed training steps file
        self.steps_file = os.path.join(self.log_dir, f"{self.agent_type}_steps_{self.timestamp}.csv")
        with open(self.steps_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'step', 'loss', 'q_value', 'reward', 'action', 'epsilon'])
        
        # Parameters tracking file
        self.params_file = os.path.join(self.log_dir, f"{self.agent_type}_params_{self.timestamp}.csv")
        with open(self.params_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'learning_rate', 'gamma', 'epsilon', 'epsilon_decay',
                           'batch_size', 'memory_size', 'update_freq'])
        
        # Task-specific metrics for meta-learning
        if self.agent_type == "meta":
            self.task_file = os.path.join(self.log_dir, f"{self.agent_type}_tasks_{self.timestamp}.csv")
            with open(self.task_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'task', 'usage_count', 'avg_reward', 'avg_q_value', 'success_rate'])
    
    def start_episode(self, episode: int, epsilon: float):
        """Start tracking a new episode"""
        self.current_episode = {
            'episode': episode,
            'score': 0,
            'steps': 0,
            'epsilon': epsilon,
            'avg_q_value': 0,
            'avg_loss': 0,
            'losses': [],
            'q_values': [],
            'actions': [],
            'rewards': [],
            'survival_time': 0,
            'asteroids_destroyed': 0,
            'accuracy': 0,
            'exploration_rate': 0
        }
    
    def log_step(self, loss: Optional[float], q_value: Optional[float], reward: float, action: int):
        """Log a single training step"""
        self.current_episode['steps'] += 1
        self.current_episode['rewards'].append(reward)
        self.current_episode['actions'].append(action)
        
        if loss is not None:
            self.current_episode['losses'].append(loss)
        if q_value is not None:
            self.current_episode['q_values'].append(q_value)
        
        # Write to detailed steps CSV
        with open(self.steps_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode['episode'],
                self.current_episode['steps'],
                loss if loss is not None else 0,
                q_value if q_value is not None else 0,
                reward,
                action,
                self.current_episode['epsilon']
            ])
    
    def log_task_metrics(self, task_name: str, usage_count: int, avg_reward: float, 
                        avg_q_value: float, success_rate: float):
        """Log task-specific metrics for meta-learning"""
        with open(self.task_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode['episode'],
                task_name,
                usage_count,
                avg_reward,
                avg_q_value,
                success_rate
            ])
    
    def end_episode(self, score: int, survival_time: int, asteroids_destroyed: int):
        """Finalize episode metrics and write to files"""
        self.current_episode['score'] = score
        self.current_episode['survival_time'] = survival_time
        self.current_episode['asteroids_destroyed'] = asteroids_destroyed
        
        # Calculate episode statistics
        if self.current_episode['losses']:
            self.current_episode['avg_loss'] = statistics.mean(self.current_episode['losses'])
        if self.current_episode['q_values']:
            self.current_episode['avg_q_value'] = statistics.mean(self.current_episode['q_values'])
        
        # Calculate action entropy (measure of exploration diversity)
        action_counts = {}
        for action in self.current_episode['actions']:
            action_counts[action] = action_counts.get(action, 0) + 1
        total_actions = len(self.current_episode['actions'])
        action_entropy = 0
        if total_actions > 0:
            for count in action_counts.values():
                p = count / total_actions
                if p > 0:
                    action_entropy -= p * math.log(p)
        
        # Calculate accuracy (shots fired vs asteroids destroyed)
        fire_actions = self.current_episode['actions'].count(3)  # Assuming action 3 is fire
        self.current_episode['accuracy'] = asteroids_destroyed / fire_actions if fire_actions > 0 else 0
        
        # Calculate exploration rate
        unique_actions = len(set(self.current_episode['actions']))
        self.current_episode['exploration_rate'] = unique_actions / 4  # Assuming 4 possible actions
        
        # Write episode summary
        with open(self.episode_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode['episode'],
                score,
                self.current_episode['steps'],
                self.current_episode['epsilon'],
                self.current_episode['avg_q_value'],
                self.current_episode['avg_loss'],
                survival_time,
                asteroids_destroyed,
                self.current_episode['accuracy'],
                self.current_episode['exploration_rate'],
                statistics.mean(self.current_episode['rewards']) if self.current_episode['rewards'] else 0,
                statistics.stdev(self.current_episode['rewards']) if len(self.current_episode['rewards']) > 1 else 0,
                action_entropy
            ])
        
        self.episode_metrics.append(self.current_episode.copy())
    
    def log_parameters(self, episode: int, params: Dict):
        """Log current training parameters"""
        with open(self.params_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                params.get('learning_rate', 0),
                params.get('gamma', 0),
                params.get('epsilon', 0),
                params.get('epsilon_decay', 0),
                params.get('batch_size', 0),
                params.get('memory_size', 0),
                params.get('update_freq', 0)
            ])
    
    def save_summary(self):
        """Save a comprehensive summary of the training session"""
        summary_file = os.path.join(self.log_dir, f"{self.agent_type}_summary_{self.timestamp}.json")
        
        # Calculate overall statistics
        all_scores = [ep['score'] for ep in self.episode_metrics]
        all_losses = [ep['avg_loss'] for ep in self.episode_metrics if ep['avg_loss'] > 0]
        all_q_values = [ep['avg_q_value'] for ep in self.episode_metrics if ep['avg_q_value'] != 0]
        
        summary = {
            'agent_type': self.agent_type,
            'timestamp': self.timestamp,
            'total_episodes': len(self.episode_metrics),
            'statistics': {
                'score': {
                    'mean': statistics.mean(all_scores) if all_scores else 0,
                    'std': statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                    'max': max(all_scores) if all_scores else 0,
                    'min': min(all_scores) if all_scores else 0
                },
                'loss': {
                    'mean': statistics.mean(all_losses) if all_losses else 0,
                    'std': statistics.stdev(all_losses) if len(all_losses) > 1 else 0
                },
                'q_value': {
                    'mean': statistics.mean(all_q_values) if all_q_values else 0,
                    'std': statistics.stdev(all_q_values) if len(all_q_values) > 1 else 0
                }
            },
            'parameters': self.parameters,
            'learning_curve': {
                'scores': all_scores,
                'moving_avg_50': self._calculate_moving_average(all_scores, 50),
                'epsilon_values': [ep['epsilon'] for ep in self.episode_metrics]
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining logs saved to: {self.log_dir}")
        print(f"Summary file: {summary_file}")
    
    def _calculate_moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average with given window size"""
        if len(values) < window:
            return values
        
        moving_avg = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            moving_avg.append(statistics.mean(values[start:i+1]))
        return moving_avg

# Define the neural network for Q-value estimation
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, fresh_start=True, logger=None, param_set="default"):
        self.state_size = state_size
        self.action_size = action_size
        
        # Load parameters from config or use defaults
        params_config = load_ai_parameters()
        if params_config and param_set in params_config.get("dqn_parameters", {}):
            params = params_config["dqn_parameters"][param_set]
            print(f"Loading DQN parameters: {param_set} - {params['description']}")
            self.learning_rate = params["learning_rate"]
            self.gamma = params["gamma"]
            self.epsilon = params["epsilon_start"] if fresh_start else params["epsilon_min"]
            self.epsilon_min = params["epsilon_min"]
            self.epsilon_decay = params["epsilon_decay"]
            self.batch_size = params["batch_size"]
            self.memory = deque(maxlen=params["memory_size"])
            self.update_frequency = params["update_frequency"]
        else:
            # Default parameters
            self.memory = deque(maxlen=10000)
            self.gamma = 0.95
            self.epsilon = 1.0 if fresh_start else 0.1
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.9998
            self.learning_rate = 0.001
            self.batch_size = 32
            self.update_frequency = 1
        
        self.episodes_trained = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_step_counter = 0
        self.logger = logger
        
        # Metrics tracking
        self.last_loss = None
        self.last_q_value = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state_tensor)
        
        # Track Q-value for logging
        self.last_q_value = torch.max(act_values).item()
        
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        self.train_step_counter += 1
        # Only update every N steps for efficiency
        if self.train_step_counter % self.update_frequency != 0:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        # Convert to numpy arrays first to avoid warning
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in minibatch])).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Track loss for logging
        self.last_loss = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Only decay epsilon occasionally to prevent too fast convergence
        if self.train_step_counter % 10 == 0 and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.epsilon = self.epsilon_min  # Set epsilon to minimum for testing
    
    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def get_training_params(self):
        """Get current training parameters for logging"""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'memory_size': len(self.memory),
            'update_freq': self.update_frequency
        }

# Task-specific network for individual game skills
class TaskSpecificNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(TaskSpecificNetwork, self).__init__()
        # Smaller network for specific tasks
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Task scheduler for dynamic task selection
class TaskScheduler:
    def __init__(self):
        self.task_priorities = {
            "survival": 0.3,
            "navigation": 0.2,
            "targeting": 0.2,
            "threat_assessment": 0.2,
            "resource_management": 0.1
        }
        self.task_performance = {task: deque(maxlen=50) for task in self.task_priorities}
        self.current_task = "survival"
        
    def select_task(self, game_state_dict):
        """Select task based on current game situation"""
        # Emergency survival mode
        if game_state_dict.get('nearest_asteroid_dist', float('inf')) < 50:
            return "survival"
        
        # If many asteroids, focus on threat assessment
        if game_state_dict.get('asteroid_count', 0) > 8:
            return "threat_assessment"
        
        # If asteroids are far, focus on targeting
        if game_state_dict.get('nearest_asteroid_dist', float('inf')) > 200:
            return "targeting"
        
        # Otherwise, use weighted random selection based on performance
        tasks = list(self.task_priorities.keys())
        weights = []
        for task in tasks:
            # Tasks with poor recent performance get higher weight
            avg_perf = np.mean(self.task_performance[task]) if self.task_performance[task] else 0.5
            weight = self.task_priorities[task] * (1.1 - avg_perf)
            weights.append(weight)
        
        return np.random.choice(tasks, p=np.array(weights)/sum(weights))
    
    def update_performance(self, task, performance):
        """Update task performance history"""
        self.task_performance[task].append(performance)

# Improved meta-learning agent with task decomposition
class ImprovedMetaLearningAgent:
    def __init__(self, state_size, action_size, fresh_start=True, logger=None):
        self.state_size = state_size + 5  # Add 5 for one-hot task encoding
        self.base_state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # Task-specific networks
        self.task_networks = {
            "navigation": TaskSpecificNetwork(self.state_size, action_size).to(self.device),
            "targeting": TaskSpecificNetwork(self.state_size, action_size).to(self.device),
            "threat_assessment": TaskSpecificNetwork(self.state_size, action_size).to(self.device),
            "resource_management": TaskSpecificNetwork(self.state_size, action_size).to(self.device),
            "survival": TaskSpecificNetwork(self.state_size, action_size).to(self.device)
        }
        
        # Separate optimizers for each task
        self.optimizers = {
            task: optim.Adam(network.parameters(), lr=0.001)
            for task, network in self.task_networks.items()
        }
        
        # Task scheduler
        self.task_scheduler = TaskScheduler()
        
        # Separate memory buffers for each task
        self.task_memories = {task: deque(maxlen=2000) for task in self.task_networks}
        
        # Exploration parameters
        self.epsilon = 1.0 if fresh_start else 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9998
        self.gamma = 0.95
        
        # Performance tracking
        self.task_usage_count = {task: 0 for task in self.task_networks}
        self.task_rewards = {task: deque(maxlen=100) for task in self.task_networks}
        self.current_task = "survival"
        
    def select_task_and_act(self, state, game_state_dict):
        """Select appropriate task and action based on game state"""
        # Select task
        self.current_task = self.task_scheduler.select_task(game_state_dict)
        self.task_usage_count[self.current_task] += 1
        
        # Add task encoding to state (one-hot)
        task_encoding = self._get_task_encoding(self.current_task)
        enhanced_state = np.concatenate([state, task_encoding])
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), self.current_task
        
        # Use task-specific network
        state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
        q_values = self.task_networks[self.current_task](state_tensor)
        return np.argmax(q_values.cpu().data.numpy()), self.current_task
    
    def _get_task_encoding(self, task):
        """Get one-hot encoding for task"""
        tasks = ["navigation", "targeting", "threat_assessment", "resource_management", "survival"]
        encoding = np.zeros(5)
        encoding[tasks.index(task)] = 1.0
        return encoding
    
    def remember(self, state, action, reward, next_state, done, task, task_reward):
        """Store experience in task-specific memory"""
        # Add task encoding to states
        task_encoding = self._get_task_encoding(task)
        enhanced_state = np.concatenate([state, task_encoding])
        enhanced_next_state = np.concatenate([next_state, task_encoding])
        
        self.task_memories[task].append(
            (enhanced_state, action, reward, enhanced_next_state, done)
        )
        self.task_rewards[task].append(task_reward)
        
        # Update task performance
        performance = min(1.0, max(0.0, (task_reward + 10) / 20))  # Normalize to [0,1]
        self.task_scheduler.update_performance(task, performance)
    
    def train(self, batch_size=32):
        """Train all task networks that have enough experience"""
        for task, memory in self.task_memories.items():
            if len(memory) >= batch_size:
                self._train_task_network(task, batch_size)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train_task_network(self, task, batch_size):
        """Train a specific task network"""
        minibatch = random.sample(self.task_memories[task], batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in minibatch])).to(self.device)
        
        current_q_values = self.task_networks[task](states).gather(1, actions.unsqueeze(1))
        next_q_values = self.task_networks[task](next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizers[task].zero_grad()
        loss.backward()
        self.optimizers[task].step()
    
    def get_performance_stats(self):
        """Get performance statistics for each task"""
        stats = {}
        for task in self.task_networks:
            stats[task] = {
                "usage_count": self.task_usage_count[task],
                "avg_reward": np.mean(self.task_rewards[task]) if self.task_rewards[task] else 0,
                "usage_percentage": self.task_usage_count[task] / max(1, sum(self.task_usage_count.values()))
            }
        return stats
    
    def save(self, filepath):
        """Save all task networks and scheduler state"""
        checkpoint = {
            'task_networks': {task: net.state_dict() for task, net in self.task_networks.items()},
            'task_scheduler': {
                'priorities': self.task_scheduler.task_priorities,
                'performance': {k: list(v) for k, v in self.task_scheduler.task_performance.items()}
            },
            'epsilon': self.epsilon,
            'task_usage': self.task_usage_count
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """Load all task networks and scheduler state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for task, state_dict in checkpoint['task_networks'].items():
            self.task_networks[task].load_state_dict(state_dict)
        
        self.task_scheduler.task_priorities = checkpoint['task_scheduler']['priorities']
        for task, perf_list in checkpoint['task_scheduler']['performance'].items():
            self.task_scheduler.task_performance[task] = deque(perf_list, maxlen=50)
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.task_usage_count = checkpoint.get('task_usage', self.task_usage_count)

# Helper functions for AI
def get_game_state():
    global player_pos, player_angle, asteroids, bullets, width, height
    
    # Convert game state to a numpy array
    state = np.array([
        player_pos[0] / width,
        player_pos[1] / height,
        math.sin(math.radians(player_angle)),
        math.cos(math.radians(player_angle)),
        len(bullets) / 5.0  # Normalized bullet count
    ])
    
    # Sort asteroids by distance to player
    asteroid_distances = []
    for asteroid in asteroids:
        dist = math.sqrt((asteroid[0] - player_pos[0])**2 + (asteroid[1] - player_pos[1])**2)
        asteroid_distances.append((dist, asteroid))
    asteroid_distances.sort(key=lambda x: x[0])
    
    # Consider up to 5 nearest asteroids with relative positions
    for i in range(5):
        if i < len(asteroid_distances):
            dist, asteroid = asteroid_distances[i]
            # Relative position to player
            rel_x = (asteroid[0] - player_pos[0]) / width
            rel_y = (asteroid[1] - player_pos[1]) / height
            state = np.append(state, [
                rel_x,
                rel_y,
                dist / 500.0  # Normalized distance
            ])
        else:
            state = np.append(state, [0, 0, 1.0])  # Far away = 1.0
    
    return state

def get_enhanced_game_state():
    """Enhanced game state with additional context for meta-learning"""
    global player_pos, player_angle, asteroids, bullets, width, height, score, level
    
    # Get basic state
    basic_state = get_game_state()
    
    # Calculate additional features
    asteroid_distances = []
    threat_levels = []
    
    for asteroid in asteroids:
        dx = asteroid[0] - player_pos[0]
        dy = asteroid[1] - player_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        # Calculate approach velocity (how fast asteroid is moving toward player)
        next_dist = math.sqrt((dx - dx/dist*asteroid_speed)**2 + (dy - dy/dist*asteroid_speed)**2)
        approach_vel = (dist - next_dist) / dist if dist > 0 else 0
        
        # Threat level based on distance, size, and approach velocity
        threat = (1.0 / max(dist, 1)) * (asteroid[2] / 50) * (1 + approach_vel)
        
        asteroid_distances.append(dist)
        threat_levels.append(threat)
    
    # Calculate game state dictionary for task scheduler
    nearest_dist = min(asteroid_distances) if asteroid_distances else float('inf')
    max_threat = max(threat_levels) if threat_levels else 0
    
    game_state_dict = {
        'nearest_asteroid_dist': nearest_dist,
        'asteroid_count': len(asteroids),
        'bullet_count': len(bullets),
        'max_threat': max_threat,
        'player_x': player_pos[0],
        'player_y': player_pos[1],
        'score': score,
        'level': level,
        'center_dist': math.sqrt((player_pos[0] - width/2)**2 + (player_pos[1] - height/2)**2)
    }
    
    # Enhanced features for the state vector
    enhanced_features = np.array([
        len(asteroids) / 15.0,  # Normalized asteroid count
        max_threat if threat_levels else 0,  # Maximum threat level
        game_state_dict['center_dist'] / (width/2),  # Distance from center
        float(len(bullets) > 0),  # Has bullets in flight
        float(nearest_dist < 100),  # In danger zone
    ])
    
    # Combine basic state with enhanced features
    full_state = np.concatenate([basic_state, enhanced_features])
    
    return full_state, game_state_dict

def calculate_task_specific_reward(task, prev_state_dict, curr_state_dict, base_reward, action_taken):
    """Calculate reward specific to each task"""
    task_reward = 0
    
    if task == "survival":
        # Reward for staying alive and avoiding close calls
        if curr_state_dict['nearest_asteroid_dist'] > 100:
            task_reward += 0.5
        if curr_state_dict['nearest_asteroid_dist'] < 50:
            task_reward -= 2.0
        # Bonus for escaping danger
        if prev_state_dict['nearest_asteroid_dist'] < 50 and curr_state_dict['nearest_asteroid_dist'] > 100:
            task_reward += 3.0
            
    elif task == "navigation":
        # Reward for efficient movement and positioning
        # Prefer center area when safe
        if curr_state_dict['center_dist'] < prev_state_dict['center_dist'] and curr_state_dict['nearest_asteroid_dist'] > 150:
            task_reward += 0.3
        # Reward for moving when needed
        if action_taken == 2:  # Forward movement
            task_reward += 0.1
            
    elif task == "targeting":
        # Reward for accurate shooting
        if action_taken == 3 and curr_state_dict['bullet_count'] > prev_state_dict['bullet_count']:
            # Check if shooting toward an asteroid
            if curr_state_dict['nearest_asteroid_dist'] < 300:
                task_reward += 1.0
        # Big reward for destroying asteroids
        if curr_state_dict['score'] > prev_state_dict['score']:
            task_reward += 5.0
            
    elif task == "threat_assessment":
        # Reward for managing multiple threats
        threat_reduction = prev_state_dict.get('max_threat', 0) - curr_state_dict.get('max_threat', 0)
        if threat_reduction > 0:
            task_reward += threat_reduction * 2
        # Reward for reducing asteroid count
        if curr_state_dict['asteroid_count'] < prev_state_dict['asteroid_count']:
            task_reward += 3.0
            
    elif task == "resource_management":
        # Reward for efficient bullet usage
        if curr_state_dict['bullet_count'] <= 3:  # Don't spam bullets
            task_reward += 0.2
        # Penalty for wasting bullets
        if action_taken == 3 and curr_state_dict['nearest_asteroid_dist'] > 400:
            task_reward -= 0.5
    
    # Add base reward
    total_reward = base_reward + task_reward
    
    return total_reward, task_reward

def perform_action(action):
    global player_angle, bullets, player_pos, player_speed, bullet_speed
    
    if action == 0:  # Turn left
        player_angle += 5
    elif action == 1:  # Turn right
        player_angle -= 5
    elif action == 2:  # Move forward
        angle_rad = math.radians(player_angle)
        player_pos[0] += math.sin(angle_rad) * player_speed
        player_pos[1] -= math.cos(angle_rad) * player_speed
        # Wrap player position
        player_pos[0] = player_pos[0] % width
        player_pos[1] = player_pos[1] % height
    elif action == 3:  # Shoot
        angle_rad = math.radians(player_angle)
        bullet_dx = math.sin(angle_rad) * bullet_speed
        bullet_dy = -math.cos(angle_rad) * bullet_speed
        bullets.append([player_pos[0], player_pos[1], bullet_dx, bullet_dy])

def update_game_state(training_level=None, max_asteroids_allowed=15, speed_multiplier=1.0):
    global asteroids, bullets, explosions, score, asteroids_destroyed, level, game_over
    global frame_count, asteroid_spawn_time, asteroid_speed

    # Update bullets
    for bullet in bullets[:]:
        bullet[0] += bullet[2]
        bullet[1] += bullet[3]
        if bullet[0] < 0 or bullet[0] > width or bullet[1] < 0 or bullet[1] > height:
            bullets.remove(bullet)

    # Update asteroids
    # Apply speed multiplier if in training mode
    speed_mult = speed_multiplier
        
    for asteroid in asteroids[:]:
        dx = player_pos[0] - asteroid[0]
        dy = player_pos[1] - asteroid[1]
        distance = math.sqrt(dx**2 + dy**2)
        if distance > 0:
            asteroid[0] += (dx / distance) * (asteroid_speed * speed_mult + level * 0.2) * 0.9 + random.uniform(-0.5, 0.5)
            asteroid[1] += (dy / distance) * (asteroid_speed * speed_mult + level * 0.2) * 0.9 + random.uniform(-0.5, 0.5)
        asteroid[0] = asteroid[0] % width
        asteroid[1] = asteroid[1] % height

        # Check for collision with player
        if check_collision(player_pos, asteroid[:2], asteroid[2]):
            game_over = True

        # Check for collision with bullets
        for bullet in bullets[:]:
            if check_collision(bullet[:2], asteroid[:2], asteroid[2]):
                if asteroid in asteroids:
                    asteroids.remove(asteroid)
                if bullet in bullets:
                    bullets.remove(bullet)
                explosions.append([asteroid[0], asteroid[1], asteroid[2], 0])
                score += 10
                asteroids_destroyed += 1
                break

    # Spawn new asteroids (limited by training level)
    if frame_count % max(60, asteroid_spawn_time - level * 5) == 0:
        if training_level is None or len(asteroids) < max_asteroids_allowed:
            asteroids.append(spawn_asteroid())

    # Update explosions
    for explosion in explosions[:]:
        explosion[3] += 1
        if explosion[3] >= 3:
            explosions.remove(explosion)

    # Level up
    if asteroids_destroyed >= 10:
        level += 1
        asteroids_destroyed = 0

def get_reward(training_level=None, level_complete=False):
    global game_over, score, asteroids, player_pos, asteroids_destroyed
    
    reward = 0
    
    # Level completion bonus
    if level_complete:
        return 50  # Big reward for completing level objectives
    
    if game_over and not level_complete:
        reward = -10  # Moderate death penalty
    else:
        # Survival reward
        reward = 0.01
        
        # Progressive rewards based on training level
        if training_level is not None:
            # Early levels reward survival more
            if training_level < 2:
                reward += 0.05
            # Later levels reward aggression more
            elif training_level >= 3:
                reward -= len(asteroids) * 0.01  # Penalty for too many asteroids
        
        # Find nearest asteroid
        if asteroids:
            min_dist = float('inf')
            for asteroid in asteroids:
                dist = math.sqrt((asteroid[0] - player_pos[0])**2 + (asteroid[1] - player_pos[1])**2)
                min_dist = min(min_dist, dist)
            
            # Reward for keeping safe distance
            if min_dist > 100:
                reward += 0.1
            elif min_dist < 50:
                reward -= 0.5  # Penalty for being too close
        
        # Bonus for having bullets in flight (encouraging shooting)
        if len(bullets) > 0:
            reward += 0.02
    
    return reward

# Training levels configuration - 15 levels with progressive difficulty
TRAINING_LEVELS = [
    # Basic skill development (Levels 1-5)
    {"level": 1, "time_limit": 180, "score_target": 20, "max_asteroids": 3, "asteroid_speed": 0.3, "focus": "survival"},
    {"level": 2, "time_limit": 240, "score_target": 40, "max_asteroids": 4, "asteroid_speed": 0.4, "focus": "navigation"},
    {"level": 3, "time_limit": 300, "score_target": 60, "max_asteroids": 5, "asteroid_speed": 0.5, "focus": "targeting"},
    {"level": 4, "time_limit": 360, "score_target": 80, "max_asteroids": 6, "asteroid_speed": 0.6, "focus": "threat_assessment"},
    {"level": 5, "time_limit": 420, "score_target": 100, "max_asteroids": 7, "asteroid_speed": 0.7, "focus": "resource_management"},
    
    # Combined skills and strategy (Levels 6-10)
    {"level": 6, "time_limit": 480, "score_target": 130, "max_asteroids": 8, "asteroid_speed": 0.8, "focus": "mixed"},
    {"level": 7, "time_limit": 540, "score_target": 160, "max_asteroids": 9, "asteroid_speed": 0.9, "focus": "mixed"},
    {"level": 8, "time_limit": 600, "score_target": 200, "max_asteroids": 10, "asteroid_speed": 1.0, "focus": "mixed"},
    {"level": 9, "time_limit": 660, "score_target": 250, "max_asteroids": 11, "asteroid_speed": 1.1, "focus": "mixed"},
    {"level": 10, "time_limit": 720, "score_target": 300, "max_asteroids": 12, "asteroid_speed": 1.2, "focus": "mixed"},
    
    # Advanced scenarios and mastery (Levels 11-15)
    {"level": 11, "time_limit": 600, "score_target": 350, "max_asteroids": 13, "asteroid_speed": 1.3, "focus": "mastery"},
    {"level": 12, "time_limit": 600, "score_target": 400, "max_asteroids": 14, "asteroid_speed": 1.4, "focus": "mastery"},
    {"level": 13, "time_limit": 600, "score_target": 450, "max_asteroids": 15, "asteroid_speed": 1.5, "focus": "mastery"},
    {"level": 14, "time_limit": 600, "score_target": 500, "max_asteroids": 16, "asteroid_speed": 1.6, "focus": "mastery"},
    {"level": 15, "time_limit": 0, "score_target": 0, "max_asteroids": 20, "asteroid_speed": 1.8, "focus": "unlimited"}  # Ultimate challenge
]

# Modify the game loop to work with the AI agent
def run_game_with_ai(agent, render=False, max_frames=2000, frame_skip=2, training_level=None):
    """Original game loop for standard DQN agent"""
    global player_pos, player_angle, bullets, asteroids, explosions, level, score, asteroids_destroyed, game_over, frame_count
    
    # Reset game state
    player_pos = [width // 2, height // 2]
    player_angle = 0
    bullets = []
    asteroids = []
    explosions = []
    level = 1
    score = 0
    asteroids_destroyed = 0
    game_over = False
    frame_count = 0
    clock = pygame.time.Clock() if render else None
    action = 0  # Initial action
    
    # Apply training level settings if specified
    if training_level is not None and 0 <= training_level < len(TRAINING_LEVELS):
        level_config = TRAINING_LEVELS[training_level]
        time_limit = level_config["time_limit"]
        score_target = level_config["score_target"]
        max_asteroids_allowed = level_config["max_asteroids"]
        speed_multiplier = level_config["asteroid_speed"]
    else:
        time_limit = 0  # No limit
        score_target = 0  # No target
        max_asteroids_allowed = 15
        speed_multiplier = 1.0
    
    level_complete = False
    time_frames = time_limit * 30 if time_limit > 0 else max_frames  # 30 fps assumed

    while not game_over and not level_complete and frame_count < time_frames:
        # Handle pygame events if rendering
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return score
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_m:
                        return score

        # AI makes decision every frame_skip frames
        if frame_count % frame_skip == 0:
            # Get game state
            state = get_game_state()
            # Choose action
            action = agent.act(state)

        # Store previous score for reward calculation
        prev_score = score

        # Perform action every frame (but same action for frame_skip frames)
        perform_action(action)

        # Update game state
        update_game_state(training_level, max_asteroids_allowed, speed_multiplier)

        # Only process learning every frame_skip frames
        if frame_count % frame_skip == 0:
            # Get new state and reward
            next_state = get_game_state()
            
            # Check level completion first
            level_complete = False
            if training_level is not None and score_target > 0 and score >= score_target:
                level_complete = True
                game_over = True  # End episode on level completion
            
            reward = get_reward(training_level, level_complete)
            # Add score-based reward
            if score > prev_score:
                reward += 5  # Big reward for destroying asteroid

            # Remember the transition
            agent.remember(state, action, reward, next_state, game_over)

            # Train the agent with larger batch
            if len(agent.memory) > 128:
                agent.replay(64)
            
            # Log step if logger exists
            if hasattr(agent, 'logger') and agent.logger is not None:
                agent.logger.log_step(agent.last_loss, agent.last_q_value, reward, action)

        # Render if requested
        if render:
            render_game(is_ai_playing=True)
            clock.tick(60)

        frame_count += 1

    # Return score and whether level was completed
    if training_level is not None:
        return score, level_complete
    return score

# New game loop for meta-learning agent
def run_game_with_meta_agent(agent, render=False, max_frames=2000, frame_skip=2, training_level=None):
    """Game loop specifically for ImprovedMetaLearningAgent with task decomposition"""
    global player_pos, player_angle, bullets, asteroids, explosions, level, score, asteroids_destroyed, game_over, frame_count
    
    # Reset game state
    player_pos = [width // 2, height // 2]
    player_angle = 0
    bullets = []
    asteroids = []
    explosions = []
    level = 1
    score = 0
    asteroids_destroyed = 0
    game_over = False
    frame_count = 0
    clock = pygame.time.Clock() if render else None
    
    # Apply training level settings if specified
    if training_level is not None and 0 <= training_level < len(TRAINING_LEVELS):
        level_config = TRAINING_LEVELS[training_level]
        time_limit = level_config["time_limit"]
        score_target = level_config["score_target"]
        max_asteroids_allowed = level_config["max_asteroids"]
        speed_multiplier = level_config["asteroid_speed"]
        level_focus = level_config.get("focus", "mixed")
    else:
        time_limit = 0
        score_target = 0
        max_asteroids_allowed = 15
        speed_multiplier = 1.0
        level_focus = "mixed"
    
    level_complete = False
    time_frames = time_limit * 30 if time_limit > 0 else max_frames
    
    # Track previous state for reward calculation
    prev_state_dict = None
    current_task = "survival"
    action = 0

    while not game_over and not level_complete and frame_count < time_frames:
        # Handle pygame events if rendering
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return score
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_m:
                        return score

        # Meta-agent makes decision every frame_skip frames
        if frame_count % frame_skip == 0:
            # Get enhanced game state
            state, curr_state_dict = get_enhanced_game_state()
            
            # Meta-agent selects task and action
            action, current_task = agent.select_task_and_act(state, curr_state_dict)

        # Store previous score for reward calculation
        prev_score = score

        # Perform action every frame
        perform_action(action)

        # Update game state
        update_game_state(training_level, max_asteroids_allowed, speed_multiplier)

        # Only process learning every frame_skip frames
        if frame_count % frame_skip == 0:
            # Get new state
            next_state, next_state_dict = get_enhanced_game_state()
            
            # Check level completion
            level_complete = False
            if training_level is not None and score_target > 0 and score >= score_target:
                level_complete = True
                game_over = True
            
            # Calculate base reward
            base_reward = get_reward(training_level, level_complete)
            
            # Add score-based reward
            if score > prev_score:
                base_reward += 5
            
            # Calculate task-specific reward
            if prev_state_dict is not None:
                total_reward, task_reward = calculate_task_specific_reward(
                    current_task, prev_state_dict, curr_state_dict, base_reward, action
                )
            else:
                total_reward, task_reward = base_reward, 0
            
            # Remember the transition with task info
            agent.remember(state, action, total_reward, next_state, game_over, current_task, task_reward)
            
            # Train the agent
            agent.train(batch_size=32)
            
            # Update previous state
            prev_state_dict = curr_state_dict.copy()

        # Render if requested
        if render:
            render_game(is_ai_playing=True, current_task=current_task)
            clock.tick(60)

        frame_count += 1

    # Return score, level completion, and task performance stats
    if training_level is not None:
        return score, level_complete, agent.get_performance_stats()
    return score

# Rendering functions
def draw_player(pos, angle):
    angle_rad = math.radians(angle)
    points = []
    # Triangle shape
    points.append((pos[0] + player_size * math.sin(angle_rad), 
                   pos[1] - player_size * math.cos(angle_rad)))
    points.append((pos[0] + player_size * math.sin(angle_rad + 2.3), 
                   pos[1] - player_size * math.cos(angle_rad + 2.3)))
    points.append((pos[0] + player_size * math.sin(angle_rad - 2.3), 
                   pos[1] - player_size * math.cos(angle_rad - 2.3)))
    pygame.draw.polygon(screen, GREEN, points)

def draw_asteroid(pos, size, color, pulse):
    pulsed_size = int(size + pulse * 2)
    points = []
    for i in range(8):
        angle = i * (360 / 8) + random.randint(-20, 20)
        x = pos[0] + pulsed_size * math.cos(math.radians(angle))
        y = pos[1] + pulsed_size * math.sin(math.radians(angle))
        points.append((x, y))
    pygame.draw.polygon(screen, color, points)

def draw_explosion(pos, size, frame):
    explosion_size = int(size + frame * 5)
    if explosion_size > 0:
        pygame.draw.circle(screen, ORANGE, (int(pos[0]), int(pos[1])), explosion_size, 2)

def draw_stars():
    for star in stars:
        pygame.draw.circle(screen, WHITE, star, 1)

def move_stars():
    global stars
    stars = [(x, (y + 1) % height) for x, y in stars]

def draw_score_and_level():
    score_text = font.render(f"Score: {score}", True, WHITE)
    level_text = font.render(f"Level: {level}", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(level_text, (width - 120, 10))

def render_game(is_ai_playing=False, current_task=None):
    screen.fill(BLACK)
    draw_stars()
    if not game_over:
        draw_player(player_pos, player_angle)
    for bullet in bullets:
        pygame.draw.circle(screen, WHITE, (int(bullet[0]), int(bullet[1])), 2)
    for asteroid in asteroids:
        draw_asteroid(asteroid[:2], asteroid[2], asteroid[3], asteroid[4])
    for explosion in explosions:
        draw_explosion(explosion[:2], explosion[2], explosion[3])
    draw_score_and_level()
    
    if is_ai_playing:
        ai_text = font.render("AI Playing", True, YELLOW)
        screen.blit(ai_text, (10, 50))
        
        # Display current task if meta-learning
        if current_task:
            task_colors = {
                "survival": RED,
                "navigation": BLUE,
                "targeting": GREEN,
                "threat_assessment": ORANGE,
                "resource_management": YELLOW
            }
            task_color = task_colors.get(current_task, WHITE)
            task_text = pygame.font.Font(None, 30).render(f"Task: {current_task.replace('_', ' ').title()}", True, task_color)
            screen.blit(task_text, (10, 80))
            
        speed_text = pygame.font.Font(None, 24).render("2x Frame Skip Active", True, GREEN)
        screen.blit(speed_text, (10, 110 if current_task else 80))
        esc_text = pygame.font.Font(None, 24).render("Press ESC or M to return to menu", True, (150, 150, 150))
        screen.blit(esc_text, (10, 135 if current_task else 105))
    
    if game_over:
        game_over_text = font.render("GAME OVER", True, RED)
        screen.blit(game_over_text, (width // 2 - game_over_text.get_width() // 2, height // 2))
    
    pygame.display.flip()

# Load/Save highscores
def load_highscores():
    global highscores
    if os.path.exists(highscore_file):
        with open(highscore_file, "r") as f:
            highscores = json.load(f)
    else:
        highscores = [0] * 5

def save_highscores():
    with open(highscore_file, "w") as f:
        json.dump(highscores, f)

def update_highscores(new_score):
    global highscores
    highscores.append(new_score)
    highscores.sort(reverse=True)
    highscores = highscores[:5]
    save_highscores()

# Train the AI
def train_ai(episodes):
    agent = DQNAgent(20, 4)  # 20 state variables, 4 possible actions
    scores = []

    for episode in range(episodes):
        score = run_game_with_ai(agent, render=False)
        scores.append(score)
        if episode % 10 == 0:
            print(f"Episode: {episode + 1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

    return agent, scores

# Training function for meta-learning agent
def train_meta_agent_headless(episodes, show_progress_every=10, fresh_start=True):
    """Train the ImprovedMetaLearningAgent with task decomposition"""
    import time
    import sys
    
    print("\nStarting Meta-Learning Agent Training...")
    print("=" * 100)
    
    # Initialize logger
    logger = TrainingLogger("meta")
    
    # Initialize meta-learning agent
    # Enhanced state size: base state (20) + enhanced features (5) = 25
    agent = ImprovedMetaLearningAgent(25, 4, fresh_start=fresh_start, logger=logger)
    
    if not fresh_start and os.path.exists("asteroids_meta_model.pth"):
        print("Loading existing meta-model...")
        agent.load("asteroids_meta_model.pth")
    
    scores = []
    level_completions = [0] * len(TRAINING_LEVELS)
    current_level = 0
    start_time = time.time()
    task_performance_history = {task: [] for task in agent.task_networks.keys()}
    
    print(f"Training for {episodes} episodes across 15 levels")
    print("Task decomposition: survival, navigation, targeting, threat_assessment, resource_management")
    print("=" * 100)
    
    for episode in range(episodes):
        episode_start = time.time()
        
        # Progressive curriculum - advance through levels
        if current_level < len(TRAINING_LEVELS) - 1:
            completions_needed = 3 + current_level  # 3, 4, 5, 6...
            if level_completions[current_level] >= completions_needed:
                current_level += 1
                print(f"\n[TARGET] Advanced to Level {current_level + 1}: {TRAINING_LEVELS[current_level]['focus']} focus\n")
        
        # Run game with meta-agent
        result = run_game_with_meta_agent(agent, render=False, training_level=current_level)
        
        if isinstance(result, tuple) and len(result) == 3:
            score, level_complete, task_stats = result
            if level_complete:
                level_completions[current_level] += 1
                
            # Track task performance
            for task, stats in task_stats.items():
                task_performance_history[task].append(stats['avg_reward'])
        else:
            score = result
            level_complete = False
            task_stats = agent.get_performance_stats()
        
        scores.append(score)
        episode_time = time.time() - episode_start
        
        # Calculate statistics
        avg_score = np.mean(scores[-min(50, len(scores)):]) if scores else 0
        max_score = max(scores) if scores else 0
        elapsed = time.time() - start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        
        # Progress display
        if (episode + 1) % show_progress_every == 0:
            progress = (episode + 1) / episodes
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "#" * filled + "-" * (bar_length - filled)
            
            print(f"\r[{bar}] Episode {episode + 1}/{episodes} ({progress*100:.1f}%) | "
                  f"L{current_level+1} | Score: {score:3d} | Avg: {avg_score:.1f} | "
                  f"e: {agent.epsilon:.3f} | {eps_per_sec:.1f} eps/s")
            
            # Show task usage statistics
            print("\n  Task Performance:")
            for task, stats in task_stats.items():
                usage_pct = stats['usage_percentage'] * 100
                avg_reward = stats['avg_reward']
                print(f"    {task:20s}: {usage_pct:5.1f}% usage, {avg_reward:6.2f} avg reward")
            
            print(f"\n  Level Progress: {level_completions}")
            print("  " + "-" * 80)
    
    # Final summary
    print("\n" + "=" * 100)
    print("Training Complete!")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final average score (last 100): {np.mean(scores[-min(100, len(scores)):]):.1f}")
    print(f"Maximum score: {max(scores)}")
    print(f"Levels completed: {sum(1 for c in level_completions if c > 0)}/15")
    
    # Task analysis
    print("\nFinal Task Analysis:")
    for task in agent.task_networks.keys():
        if task_performance_history[task]:
            avg = np.mean(task_performance_history[task][-50:])
            improvement = task_performance_history[task][-1] - task_performance_history[task][0] if len(task_performance_history[task]) > 1 else 0
            print(f"  {task}: {avg:.2f} avg performance, {improvement:+.2f} improvement")
    
    return agent, scores

# Main game function
def play_human_game():
    global player_pos, player_angle, bullets, asteroids, explosions, level, score, asteroids_destroyed, game_over, frame_count
    
    # Reset game
    player_pos = [width // 2, height // 2]
    player_angle = 0
    bullets = []
    asteroids = []
    explosions = []
    level = 1
    score = 0
    asteroids_destroyed = 0
    game_over = False
    frame_count = 0
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over:
                    angle_rad = math.radians(player_angle)
                    bullet_dx = math.sin(angle_rad) * bullet_speed
                    bullet_dy = -math.cos(angle_rad) * bullet_speed
                    bullets.append([player_pos[0], player_pos[1], bullet_dx, bullet_dy])
                if event.key == pygame.K_r and game_over:
                    # Reset game
                    player_pos = [width // 2, height // 2]
                    player_angle = 0
                    bullets = []
                    asteroids = []
                    explosions = []
                    level = 1
                    score = 0
                    asteroids_destroyed = 0
                    game_over = False
                    frame_count = 0

        if not game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                perform_action(0)  # Turn left
            if keys[pygame.K_RIGHT]:
                perform_action(1)  # Turn right
            if keys[pygame.K_UP]:
                perform_action(2)  # Move forward
            
            # Update game (no training level for human play)
            update_game_state()
            move_stars()

        # Render
        render_game()
        
        if game_over:
            restart_text = font.render("Press 'R' to restart", True, WHITE)
            menu_text = font.render("Press 'M' for menu", True, WHITE)
            screen.blit(restart_text, (width // 2 - restart_text.get_width() // 2, height // 2 + 50))
            screen.blit(menu_text, (width // 2 - menu_text.get_width() // 2, height // 2 + 90))
            
            # Update highscores
            if score > 0:
                update_highscores(score)
                score = 0  # Reset to prevent multiple updates
                
            # Check for menu key
            keys = pygame.key.get_pressed()
            if keys[pygame.K_m]:
                return False

        pygame.display.flip()
        clock.tick(60)
        frame_count += 1

    return running

# Menu system
def draw_menu():
    screen.fill(BLACK)
    draw_stars()
    
    # Title
    title_font = pygame.font.Font(None, 72)
    title_text = title_font.render("ASTEROIDS", True, WHITE)
    title_rect = title_text.get_rect(center=(width // 2, 100))
    screen.blit(title_text, title_rect)
    
    subtitle_font = pygame.font.Font(None, 36)
    subtitle_text = subtitle_font.render("AI Driven Edition", True, YELLOW)
    subtitle_rect = subtitle_text.get_rect(center=(width // 2, 150))
    screen.blit(subtitle_text, subtitle_rect)
    
    # Menu items
    menu_items = [
        ("Train Basic AI", 220),
        ("Train Meta-Learning AI", 270),
        ("Watch Basic AI", 320),
        ("Watch Meta AI", 370),
        ("Play Game", 420),
        ("Quit", 470)
    ]
    
    menu_rects = []
    for text, y in menu_items:
        # Check if AI models exist
        if text == "Watch Basic AI" and not os.path.exists("asteroids_ai_model.pth"):
            color = (100, 100, 100)  # Gray out if no model
            text += " (No Model)"
        elif text == "Watch Meta AI" and not os.path.exists("asteroids_meta_model.pth"):
            color = (100, 100, 100)  # Gray out if no model
            text += " (No Model)"
        else:
            color = WHITE
            
        item_text = font.render(text, True, color)
        item_rect = item_text.get_rect(center=(width // 2, y))
        menu_rects.append((item_rect, text))
        
        # Highlight on hover
        mouse_pos = pygame.mouse.get_pos()
        if item_rect.collidepoint(mouse_pos) and color == WHITE:
            pygame.draw.rect(screen, YELLOW, item_rect.inflate(20, 10), 2)
            
        screen.blit(item_text, item_rect)
    
    # Instructions
    inst_font = pygame.font.Font(None, 24)
    instructions = [
        "Controls: Arrow Keys to move, Space to shoot",
        "AI will learn to play through reinforcement learning"
    ]
    for i, inst in enumerate(instructions):
        inst_text = inst_font.render(inst, True, (150, 150, 150))
        inst_rect = inst_text.get_rect(center=(width // 2, 520 + i * 25))
        screen.blit(inst_text, inst_rect)
    
    pygame.display.flip()
    return menu_rects

def show_training_screen(episode, total_episodes, score, epsilon, avg_score=0, max_score=0, training_time=0):
    screen.fill(BLACK)
    draw_stars()
    
    title_text = font.render("Training AI...", True, WHITE)
    screen.blit(title_text, (width // 2 - title_text.get_width() // 2, 150))
    
    progress = episode / total_episodes
    bar_width = 400
    bar_height = 30
    bar_x = (width - bar_width) // 2
    bar_y = 200
    
    # Progress bar background
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    # Progress bar fill
    pygame.draw.rect(screen, GREEN, (bar_x, bar_y, int(bar_width * progress), bar_height))
    # Progress bar text
    percent_text = font.render(f"{int(progress * 100)}%", True, WHITE)
    screen.blit(percent_text, (bar_x + bar_width // 2 - percent_text.get_width() // 2, bar_y + 2))
    
    # Stats
    stats_font = pygame.font.Font(None, 28)
    stats = [
        f"Episode: {episode}/{total_episodes}",
        f"Current Score: {score}",
        f"Average Score (last 50): {avg_score:.1f}",
        f"Max Score: {max_score}",
        f"Exploration Rate: {epsilon:.2%}",
        f"Time Elapsed: {training_time:.1f}s",
        "",
        "Press ESC to stop and save"
    ]
    for i, stat in enumerate(stats):
        color = WHITE if stat else WHITE
        if "ESC" in stat:
            color = (150, 150, 150)
        stat_text = stats_font.render(stat, True, color)
        screen.blit(stat_text, (width // 2 - stat_text.get_width() // 2, 250 + i * 30))
    
    pygame.display.flip()

# Headless training function with simple progress
def train_ai_headless(episodes, show_progress_every=10, fresh_start=True, use_curriculum=True, param_set="default"):
    import time
    import sys
    
    # Option to start fresh or continue training
    if fresh_start and os.path.exists("asteroids_ai_model.pth"):
        print("Starting fresh training (ignoring existing model)...")
    
    # Initialize logger
    logger = TrainingLogger("dqn")
    
    agent = DQNAgent(20, 4, fresh_start=fresh_start, logger=logger, param_set=param_set)
    
    # Load existing model if not fresh start
    if not fresh_start and os.path.exists("asteroids_ai_model.pth"):
        print("Loading existing model to continue training...")
        agent.load("asteroids_ai_model.pth")
    
    # Log initial parameters
    logger.parameters = agent.get_training_params()
    logger.log_parameters(0, agent.get_training_params())
    
    scores = []
    kills = []  # Track asteroids destroyed per episode
    survival_times = []  # Track how long AI survives
    level_completions = [0] * len(TRAINING_LEVELS)  # Track completions per level
    current_level = 0
    start_time = time.time()
    best_avg = 0
    
    print(f"\nStarting headless training for {episodes} episodes...")
    print("Learning Strategy: Curriculum Learning with Progressive Levels" if use_curriculum else "Standard Training")
    print("Epsilon decay: 1.0 ? 0.01 over ~2000 episodes (very gradual)")
    if use_curriculum:
        print("\nTraining Levels:")
        for i, lvl in enumerate(TRAINING_LEVELS):
            if lvl["time_limit"] > 0:
                print(f"  Level {i+1}: {lvl['time_limit']}s time limit, {lvl['score_target']} points target, max {lvl['max_asteroids']} asteroids")
            else:
                print(f"  Level {i+1}: Unlimited play")
    print("=" * 100)
    
    for episode in range(episodes):
        episode_start = time.time()
        
        # Start episode logging
        logger.start_episode(episode, agent.epsilon)
        
        # Determine training level based on curriculum
        if use_curriculum:
            # Progress to next level after sufficient completions
            if current_level < len(TRAINING_LEVELS) - 1:
                completions_needed = 5 + current_level * 2  # 5, 7, 9, 11...
                if level_completions[current_level] >= completions_needed:
                    current_level += 1
                    print(f"\n[TARGET] Progressed to Level {current_level + 1}!\n")
            
            result = run_game_with_ai(agent, render=False, training_level=current_level)
            if isinstance(result, tuple):
                score, level_complete = result
                if level_complete:
                    level_completions[current_level] += 1
            else:
                score = result
                level_complete = False
        else:
            score = run_game_with_ai(agent, render=False)
            level_complete = False
            
        episode_time = time.time() - episode_start
        
        scores.append(score)
        kills.append(score // 10)  # Each asteroid = 10 points
        survival_times.append(episode_time)
        
        # End episode logging
        asteroids_destroyed = score // 10
        survival_frames = int(episode_time * 60)  # Approximate frames
        logger.end_episode(score, survival_frames, asteroids_destroyed)
        
        # Log parameters every 50 episodes
        if episode % 50 == 0:
            logger.log_parameters(episode, agent.get_training_params())
        
        # Calculate statistics
        avg_score = np.mean(scores[-min(50, len(scores)):]) if scores else 0
        avg_kills = np.mean(kills[-min(50, len(kills)):]) if kills else 0
        max_score = max(scores) if scores else 0
        elapsed = time.time() - start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        remaining = (episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
        
        # Track improvement
        if avg_score > best_avg:
            best_avg = avg_score
        
        # Determine learning phase
        if agent.epsilon > 0.5:
            phase = "EXPLORING"
            phase_color = "\033[93m"  # Yellow
        elif agent.epsilon > 0.1:
            phase = "LEARNING"
            phase_color = "\033[94m"  # Blue
        else:
            phase = "EXPLOITING"
            phase_color = "\033[92m"  # Green
        
        # Update counter on same line
        progress_bar_length = 20
        progress = (episode + 1) / episodes
        filled_length = int(progress_bar_length * progress)
        bar = "#" * filled_length + "-" * (progress_bar_length - filled_length)
        
        # Create status line
        if use_curriculum:
            level_str = f"L{current_level+1} "
            completions_str = f"[{level_completions[current_level]}?] "
        else:
            level_str = ""
            completions_str = ""
            
        status = (f"\r[{bar}] Episode {episode + 1}/{episodes} ({progress*100:.1f}%) | "
                 f"{level_str}{completions_str}"
                 f"Score: {score:3d} | Avg: {avg_score:5.1f} | Kills: {avg_kills:.1f} | "
                 f"Best: {best_avg:.1f} | ?: {agent.epsilon:.3f} | "
                 f"{phase_color}{phase}\033[0m | {eps_per_sec:.1f} eps/s")
        
        # Print status (overwrites same line)
        sys.stdout.write(status)
        sys.stdout.flush()
        
        # Show detailed progress every N episodes
        if (episode + 1) % show_progress_every == 0 and show_progress_every > 1:
            print()  # New line for detailed update
            print(f"  ?? Last 10 scores: {scores[-10:]}")
            print(f"  ?? Asteroids destroyed: {kills[-10:]}")
            if use_curriculum:
                print(f"  ?? Level completions: {level_completions}")
            print(f"  ?? Memory size: {len(agent.memory)} experiences")
            
            # Show milestone messages
            if episode + 1 == 100:
                print("  ? Milestone: 100 episodes - AI should be learning basic survival")
            elif episode + 1 == 300:
                print("  ? Milestone: 300 episodes - AI should be improving accuracy")
            elif episode + 1 == 500:
                print("  ? Milestone: 500 episodes - AI should be developing strategy")
    
    print()  # Final newline
    total_time = time.time() - start_time
    print("=" * 100)
    print(f"Training complete in {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"\nFinal Statistics:")
    print(f"  ? Average score (last 100): {np.mean(scores[-min(100, len(scores)):]):.1f}")
    print(f"  ? Maximum score achieved: {max(scores)}")
    print(f"  ? Best average (50 episodes): {best_avg:.1f}")
    print(f"  ? Total asteroids destroyed: {sum(kills)}")
    print(f"  ? Average training speed: {len(scores)/total_time:.1f} episodes/second")
    print(f"  ? Final exploration rate: {agent.epsilon:.3f}")
    
    # Performance assessment
    final_avg = np.mean(scores[-min(100, len(scores)):])
    if final_avg < 20:
        print("\n[WARNING]  Performance: POOR - AI needs more training or parameter tuning")
    elif final_avg < 50:
        print("\n?  Performance: DECENT - AI learned basic gameplay")
    elif final_avg < 100:
        print("\n?  Performance: GOOD - AI is competent at the game")
    else:
        print("\n[STAR] Performance: EXCELLENT - AI mastered the game!")
    
    # Save training logs summary
    logger.save_summary()
    
    return agent, scores

# Training function for meta-learning agent
def train_meta_agent_headless(episodes, show_progress_every=10, fresh_start=True):
    """Train the ImprovedMetaLearningAgent with task decomposition"""
    import time
    import sys
    
    print("\nStarting Meta-Learning Agent Training...")
    print("=" * 100)
    
    # Initialize logger
    logger = TrainingLogger("meta")
    
    # Initialize meta-learning agent
    # Enhanced state size: base state (20) + enhanced features (5) = 25
    agent = ImprovedMetaLearningAgent(25, 4, fresh_start=fresh_start, logger=logger)
    
    if not fresh_start and os.path.exists("asteroids_meta_model.pth"):
        print("Loading existing meta-model...")
        agent.load("asteroids_meta_model.pth")
    
    scores = []
    level_completions = [0] * len(TRAINING_LEVELS)
    current_level = 0
    start_time = time.time()
    task_performance_history = {task: [] for task in agent.task_networks.keys()}
    
    print(f"Training for {episodes} episodes across 15 levels")
    print("Task decomposition: survival, navigation, targeting, threat_assessment, resource_management")
    print("=" * 100)
    
    for episode in range(episodes):
        episode_start = time.time()
        
        # Progressive curriculum - advance through levels
        if current_level < len(TRAINING_LEVELS) - 1:
            completions_needed = 3 + current_level  # 3, 4, 5, 6...
            if level_completions[current_level] >= completions_needed:
                current_level += 1
                print(f"\n[TARGET] Advanced to Level {current_level + 1}: {TRAINING_LEVELS[current_level]['focus']} focus\n")
        
        # Run game with meta-agent
        result = run_game_with_meta_agent(agent, render=False, training_level=current_level)
        
        if isinstance(result, tuple) and len(result) == 3:
            score, level_complete, task_stats = result
            if level_complete:
                level_completions[current_level] += 1
                
            # Track task performance
            for task, stats in task_stats.items():
                task_performance_history[task].append(stats['avg_reward'])
        else:
            score = result
            level_complete = False
            task_stats = agent.get_performance_stats()
        
        scores.append(score)
        episode_time = time.time() - episode_start
        
        # Calculate statistics
        avg_score = np.mean(scores[-min(50, len(scores)):]) if scores else 0
        max_score = max(scores) if scores else 0
        elapsed = time.time() - start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        
        # Progress display
        if (episode + 1) % show_progress_every == 0:
            progress = (episode + 1) / episodes
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "#" * filled + "-" * (bar_length - filled)
            
            print(f"\r[{bar}] Episode {episode + 1}/{episodes} ({progress*100:.1f}%) | "
                  f"L{current_level+1} | Score: {score:3d} | Avg: {avg_score:.1f} | "
                  f"e: {agent.epsilon:.3f} | {eps_per_sec:.1f} eps/s")
            
            # Show task usage statistics
            print("\n  Task Performance:")
            for task, stats in task_stats.items():
                usage_pct = stats['usage_percentage'] * 100
                avg_reward = stats['avg_reward']
                print(f"    {task:20s}: {usage_pct:5.1f}% usage, {avg_reward:6.2f} avg reward")
            
            print(f"\n  Level Progress: {level_completions}")
            print("  " + "-" * 80)
    
    # Final summary
    print("\n" + "=" * 100)
    print("Training Complete!")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final average score (last 100): {np.mean(scores[-min(100, len(scores)):]):.1f}")
    print(f"Maximum score: {max(scores)}")
    print(f"Levels completed: {sum(1 for c in level_completions if c > 0)}/15")
    
    # Task analysis
    print("\nFinal Task Analysis:")
    for task in agent.task_networks.keys():
        if task_performance_history[task]:
            avg = np.mean(task_performance_history[task][-50:])
            improvement = task_performance_history[task][-1] - task_performance_history[task][0] if len(task_performance_history[task]) > 1 else 0
            print(f"  {task}: {avg:.2f} avg performance, {improvement:+.2f} improvement")
    
    return agent, scores

# Modified GUI training function
def train_ai_with_gui(episodes, fresh_start=True):
    import time
    agent = DQNAgent(20, 4, fresh_start=fresh_start)
    
    if not fresh_start and os.path.exists("asteroids_ai_model.pth"):
        agent.load("asteroids_ai_model.pth")
    
    scores = []
    start_time = time.time()
    
    for episode in range(episodes):
        # Check for ESC key to cancel training
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, scores
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print(f"\nTraining stopped at episode {episode + 1}")
                    return agent, scores

# Training function for meta-learning agent
def train_meta_agent_headless(episodes, show_progress_every=10, fresh_start=True):
    """Train the ImprovedMetaLearningAgent with task decomposition"""
    import time
    import sys
    
    print("\nStarting Meta-Learning Agent Training...")
    print("=" * 100)
    
    # Initialize logger
    logger = TrainingLogger("meta")
    
    # Initialize meta-learning agent
    # Enhanced state size: base state (20) + enhanced features (5) = 25
    agent = ImprovedMetaLearningAgent(25, 4, fresh_start=fresh_start, logger=logger)
    
    if not fresh_start and os.path.exists("asteroids_meta_model.pth"):
        print("Loading existing meta-model...")
        agent.load("asteroids_meta_model.pth")
    
    scores = []
    level_completions = [0] * len(TRAINING_LEVELS)
    current_level = 0
    start_time = time.time()
    task_performance_history = {task: [] for task in agent.task_networks.keys()}
    
    print(f"Training for {episodes} episodes across 15 levels")
    print("Task decomposition: survival, navigation, targeting, threat_assessment, resource_management")
    print("=" * 100)
    
    for episode in range(episodes):
        episode_start = time.time()
        
        # Progressive curriculum - advance through levels
        if current_level < len(TRAINING_LEVELS) - 1:
            completions_needed = 3 + current_level  # 3, 4, 5, 6...
            if level_completions[current_level] >= completions_needed:
                current_level += 1
                print(f"\n[TARGET] Advanced to Level {current_level + 1}: {TRAINING_LEVELS[current_level]['focus']} focus\n")
        
        # Run game with meta-agent
        result = run_game_with_meta_agent(agent, render=False, training_level=current_level)
        
        if isinstance(result, tuple) and len(result) == 3:
            score, level_complete, task_stats = result
            if level_complete:
                level_completions[current_level] += 1
                
            # Track task performance
            for task, stats in task_stats.items():
                task_performance_history[task].append(stats['avg_reward'])
        else:
            score = result
            level_complete = False
            task_stats = agent.get_performance_stats()
        
        scores.append(score)
        episode_time = time.time() - episode_start
        
        # Calculate statistics
        avg_score = np.mean(scores[-min(50, len(scores)):]) if scores else 0
        max_score = max(scores) if scores else 0
        elapsed = time.time() - start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        
        # Progress display
        if (episode + 1) % show_progress_every == 0:
            progress = (episode + 1) / episodes
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "#" * filled + "-" * (bar_length - filled)
            
            print(f"\r[{bar}] Episode {episode + 1}/{episodes} ({progress*100:.1f}%) | "
                  f"L{current_level+1} | Score: {score:3d} | Avg: {avg_score:.1f} | "
                  f"e: {agent.epsilon:.3f} | {eps_per_sec:.1f} eps/s")
            
            # Show task usage statistics
            print("\n  Task Performance:")
            for task, stats in task_stats.items():
                usage_pct = stats['usage_percentage'] * 100
                avg_reward = stats['avg_reward']
                print(f"    {task:20s}: {usage_pct:5.1f}% usage, {avg_reward:6.2f} avg reward")
            
            print(f"\n  Level Progress: {level_completions}")
            print("  " + "-" * 80)
    
    # Final summary
    print("\n" + "=" * 100)
    print("Training Complete!")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final average score (last 100): {np.mean(scores[-min(100, len(scores)):]):.1f}")
    print(f"Maximum score: {max(scores)}")
    print(f"Levels completed: {sum(1 for c in level_completions if c > 0)}/15")
    
    # Task analysis
    print("\nFinal Task Analysis:")
    for task in agent.task_networks.keys():
        if task_performance_history[task]:
            avg = np.mean(task_performance_history[task][-50:])
            improvement = task_performance_history[task][-1] - task_performance_history[task][0] if len(task_performance_history[task]) > 1 else 0
            print(f"  {task}: {avg:.2f} avg performance, {improvement:+.2f} improvement")
    
    return agent, scores

# Training function for meta-learning agent
def train_meta_agent_headless(episodes, show_progress_every=10, fresh_start=True):
    """Train the ImprovedMetaLearningAgent with task decomposition"""
    import time
    import sys
    
    print("\nStarting Meta-Learning Agent Training...")
    print("=" * 100)
    
    # Initialize logger
    logger = TrainingLogger("meta")
    
    # Initialize meta-learning agent
    # Enhanced state size: base state (20) + enhanced features (5) = 25
    agent = ImprovedMetaLearningAgent(25, 4, fresh_start=fresh_start, logger=logger)
    
    if not fresh_start and os.path.exists("asteroids_meta_model.pth"):
        print("Loading existing meta-model...")
        agent.load("asteroids_meta_model.pth")
    
    scores = []
    level_completions = [0] * len(TRAINING_LEVELS)
    current_level = 0
    start_time = time.time()
    task_performance_history = {task: [] for task in agent.task_networks.keys()}
    
    print(f"Training for {episodes} episodes across 15 levels")
    print("Task decomposition: survival, navigation, targeting, threat_assessment, resource_management")
    print("=" * 100)
    
    for episode in range(episodes):
        episode_start = time.time()
        
        # Progressive curriculum - advance through levels
        if current_level < len(TRAINING_LEVELS) - 1:
            completions_needed = 3 + current_level  # 3, 4, 5, 6...
            if level_completions[current_level] >= completions_needed:
                current_level += 1
                print(f"\n[TARGET] Advanced to Level {current_level + 1}: {TRAINING_LEVELS[current_level]['focus']} focus\n")
        
        # Run game with meta-agent
        result = run_game_with_meta_agent(agent, render=False, training_level=current_level)
        
        if isinstance(result, tuple) and len(result) == 3:
            score, level_complete, task_stats = result
            if level_complete:
                level_completions[current_level] += 1
                
            # Track task performance
            for task, stats in task_stats.items():
                task_performance_history[task].append(stats['avg_reward'])
        else:
            score = result
            level_complete = False
            task_stats = agent.get_performance_stats()
        
        scores.append(score)
        episode_time = time.time() - episode_start
        
        # Calculate statistics
        avg_score = np.mean(scores[-min(50, len(scores)):]) if scores else 0
        max_score = max(scores) if scores else 0
        elapsed = time.time() - start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        
        # Progress display
        if (episode + 1) % show_progress_every == 0:
            progress = (episode + 1) / episodes
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "#" * filled + "-" * (bar_length - filled)
            
            print(f"\r[{bar}] Episode {episode + 1}/{episodes} ({progress*100:.1f}%) | "
                  f"L{current_level+1} | Score: {score:3d} | Avg: {avg_score:.1f} | "
                  f"e: {agent.epsilon:.3f} | {eps_per_sec:.1f} eps/s")
            
            # Show task usage statistics
            print("\n  Task Performance:")
            for task, stats in task_stats.items():
                usage_pct = stats['usage_percentage'] * 100
                avg_reward = stats['avg_reward']
                print(f"    {task:20s}: {usage_pct:5.1f}% usage, {avg_reward:6.2f} avg reward")
            
            print(f"\n  Level Progress: {level_completions}")
            print("  " + "-" * 80)
    
    # Final summary
    print("\n" + "=" * 100)
    print("Training Complete!")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final average score (last 100): {np.mean(scores[-min(100, len(scores)):]):.1f}")
    print(f"Maximum score: {max(scores)}")
    print(f"Levels completed: {sum(1 for c in level_completions if c > 0)}/15")
    
    # Task analysis
    print("\nFinal Task Analysis:")
    for task in agent.task_networks.keys():
        if task_performance_history[task]:
            avg = np.mean(task_performance_history[task][-50:])
            improvement = task_performance_history[task][-1] - task_performance_history[task][0] if len(task_performance_history[task]) > 1 else 0
            print(f"  {task}: {avg:.2f} avg performance, {improvement:+.2f} improvement")
    
    return agent, scores

# Main menu loop
def main_menu():
    clock = pygame.time.Clock()
    running = True
    
    while running:
        menu_rects = draw_menu()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                for rect, text in menu_rects:
                    if rect.collidepoint(mouse_pos):
                        if "Train Basic AI" in text:
                            # Ask for number of episodes
                            screen.fill(BLACK)
                            prompt_text = font.render("Enter number of episodes (default 1000):", True, WHITE)
                            screen.blit(prompt_text, (width // 2 - prompt_text.get_width() // 2, height // 2 - 50))
                            input_text = font.render("Press 1-5 for: 1=500, 2=1000, 3=2000, 4=5000, 5=10000", True, YELLOW)
                            screen.blit(input_text, (width // 2 - input_text.get_width() // 2, height // 2))
                            pygame.display.flip()
                            
                            episodes_map = {'1': 500, '2': 1000, '3': 2000, '4': 5000, '5': 10000}
                            episodes = 1000  # default
                            
                            waiting = True
                            while waiting:
                                for evt in pygame.event.get():
                                    if evt.type == pygame.KEYDOWN:
                                        if evt.key == pygame.K_1: episodes = 500; waiting = False
                                        elif evt.key == pygame.K_2: episodes = 1000; waiting = False
                                        elif evt.key == pygame.K_3: episodes = 2000; waiting = False
                                        elif evt.key == pygame.K_4: episodes = 5000; waiting = False
                                        elif evt.key == pygame.K_5: episodes = 10000; waiting = False
                                        elif evt.key == pygame.K_RETURN: waiting = False
                                        elif evt.key == pygame.K_ESCAPE: waiting = False; episodes = 0
                            
                            if episodes > 0:
                                # Ask for parameter set
                                screen.fill(BLACK)
                                param_text = font.render("Select Parameter Set:", True, WHITE)
                                screen.blit(param_text, (width // 2 - param_text.get_width() // 2, height // 2 - 100))
                                
                                options = [
                                    "1: Default (slow epsilon decay)",
                                    "2: Fast Learning (too fast - not recommended)",
                                    "3: Balanced Learning (recommended)",
                                    "4: Stable Learning (conservative)",
                                    "5: Exploration Focused"
                                ]
                                
                                for i, opt in enumerate(options):
                                    opt_text = font.render(opt, True, YELLOW if i == 2 else WHITE)
                                    screen.blit(opt_text, (width // 2 - opt_text.get_width() // 2, height // 2 - 20 + i * 30))
                                pygame.display.flip()
                                
                                param_set = "default"
                                waiting = True
                                while waiting:
                                    for evt in pygame.event.get():
                                        if evt.type == pygame.KEYDOWN:
                                            if evt.key == pygame.K_1: param_set = "default"; waiting = False
                                            elif evt.key == pygame.K_2: param_set = "fast_learning"; waiting = False
                                            elif evt.key == pygame.K_3: param_set = "balanced_learning"; waiting = False
                                            elif evt.key == pygame.K_4: param_set = "stable_learning"; waiting = False
                                            elif evt.key == pygame.K_5: param_set = "exploration_focused"; waiting = False
                                            elif evt.key == pygame.K_ESCAPE: waiting = False; episodes = 0
                                
                                if episodes > 0:
                                    # Always start fresh for better results
                                    fresh_start = True
                                    
                                    # Show training info
                                    screen.fill(BLACK)
                                    info_lines = [
                                        "Starting AI Training",
                                        "",
                                        f"Episodes: {episodes}",
                                        f"Parameter Set: {param_set}",
                                        "Strategy: Deep Q-Learning with Logging",
                                        "",
                                        "Training will begin in 3 seconds..."
                                    ]
                                    
                                    for i, line in enumerate(info_lines):
                                        color = WHITE if i != 0 else YELLOW
                                        text = font.render(line, True, color)
                                        screen.blit(text, (width // 2 - text.get_width() // 2, 200 + i * 40))
                                    pygame.display.flip()
                                    pygame.time.wait(3000)
                                    
                                    # Minimize window and run headless training
                                    pygame.display.iconify()
                                    agent, scores = train_ai_headless(episodes, fresh_start=fresh_start, param_set=param_set)
                                torch.save(agent.model.state_dict(), "asteroids_ai_model.pth")
                                
                                # Restore window and show completion
                                pygame.display.set_mode((width, height))
                                screen.fill(BLACK)
                                msg = font.render("Training Complete!", True, GREEN)
                                screen.blit(msg, (width // 2 - msg.get_width() // 2, height // 2 - 40))
                                if len(scores) >= 100:
                                    avg_msg = font.render(f"Avg Score (last 100): {np.mean(scores[-100:]):.2f}", True, WHITE)
                                    screen.blit(avg_msg, (width // 2 - avg_msg.get_width() // 2, height // 2))
                                    max_msg = font.render(f"Max Score: {max(scores)}", True, WHITE)
                                    screen.blit(max_msg, (width // 2 - max_msg.get_width() // 2, height // 2 + 40))
                                pygame.display.flip()
                                pygame.time.wait(3000)
                        
                        elif "Train Meta-Learning AI" in text:
                            # Meta-learning AI training
                            screen.fill(BLACK)
                            prompt_text = font.render("Meta-Learning AI Training", True, YELLOW)
                            screen.blit(prompt_text, (width // 2 - prompt_text.get_width() // 2, height // 2 - 100))
                            info_lines = [
                                "15 Progressive Levels",
                                "5 Task-Specific Networks",
                                "Task Decomposition: Survival, Navigation, Targeting, etc.",
                                "",
                                "Press 1-3 for episodes: 1=500, 2=1000, 3=2000"
                            ]
                            for i, line in enumerate(info_lines):
                                line_text = font.render(line, True, WHITE)
                                screen.blit(line_text, (width // 2 - line_text.get_width() // 2, height // 2 - 50 + i * 30))
                            pygame.display.flip()
                            
                            episodes = 1000
                            waiting = True
                            while waiting:
                                for evt in pygame.event.get():
                                    if evt.type == pygame.KEYDOWN:
                                        if evt.key == pygame.K_1: episodes = 500; waiting = False
                                        elif evt.key == pygame.K_2: episodes = 1000; waiting = False
                                        elif evt.key == pygame.K_3: episodes = 2000; waiting = False
                                        elif evt.key == pygame.K_ESCAPE: waiting = False; episodes = 0
                            
                            if episodes > 0:
                                pygame.display.iconify()
                                agent, scores = train_meta_agent_headless(episodes, fresh_start=True)
                                torch.save(agent, "asteroids_meta_model.pth")
                                
                                pygame.display.set_mode((width, height))
                                screen.fill(BLACK)
                                msg = font.render("Meta-Learning Training Complete!", True, GREEN)
                                screen.blit(msg, (width // 2 - msg.get_width() // 2, height // 2 - 40))
                                if len(scores) >= 100:
                                    avg_msg = font.render(f"Avg Score (last 100): {np.mean(scores[-100:]):.2f}", True, WHITE)
                                    screen.blit(avg_msg, (width // 2 - avg_msg.get_width() // 2, height // 2))
                                pygame.display.flip()
                                pygame.time.wait(3000)
                                
                        elif "Watch Basic AI" in text and os.path.exists("asteroids_ai_model.pth"):
                            print("Loading Basic AI...")
                            agent = DQNAgent(20, 4)
                            agent.load("asteroids_ai_model.pth")
                            score = run_game_with_ai(agent, render=True)
                            # Show score
                            screen.fill(BLACK)
                            msg = font.render(f"Basic AI Score: {score}", True, WHITE)
                            screen.blit(msg, (width // 2 - msg.get_width() // 2, height // 2))
                            pygame.display.flip()
                            pygame.time.wait(2000)
                            
                        elif "Watch Meta AI" in text and os.path.exists("asteroids_meta_model.pth"):
                            print("Loading Meta-Learning AI...")
                            agent = torch.load("asteroids_meta_model.pth")
                            score = run_game_with_meta_agent(agent, render=True)
                            # Show score and task stats
                            screen.fill(BLACK)
                            msg = font.render(f"Meta AI Score: {score}", True, WHITE)
                            screen.blit(msg, (width // 2 - msg.get_width() // 2, height // 2 - 60))
                            
                            # Show task usage stats
                            stats = agent.get_performance_stats()
                            y_offset = height // 2 - 20
                            for task, task_stats in stats.items():
                                usage = task_stats['usage_percentage'] * 100
                                stat_text = font.render(f"{task}: {usage:.1f}% usage", True, YELLOW)
                                screen.blit(stat_text, (width // 2 - stat_text.get_width() // 2, y_offset))
                                y_offset += 30
                            
                            pygame.display.flip()
                            pygame.time.wait(5000)
                            
                        elif "Play Game" in text:
                            play_human_game()
                            
                                
                        elif "Quit" in text:
                            running = False
        
        clock.tick(60)
    
    return running

# Main execution
if __name__ == "__main__":
    load_highscores()
    
    try:
        main_menu()
    finally:
        pygame.quit()