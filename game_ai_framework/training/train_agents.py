"""
Training Module for Task-Specific Agents
Trains agents on Asteroids game with proper logging and saving
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datetime import datetime
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from games.asteroids import AsteroidsGame
from agents.task_agents import create_task_agent
from agents.meta_agent import MetaLearningAgent
from core.base_agent import TaskSpecificAgent


class TrainingLogger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            'episodes': [],
            'scores': [],
            'losses': [],
            'task_performance': {}
        }
    
    def log_episode(self, episode: int, score: float, steps: int, task_metrics: Dict[str, float]):
        """Log episode results"""
        self.metrics['episodes'].append({
            'episode': episode,
            'score': score,
            'steps': steps,
            'timestamp': datetime.now().isoformat()
        })
        
        for task, value in task_metrics.items():
            if task not in self.metrics['task_performance']:
                self.metrics['task_performance'][task] = []
            self.metrics['task_performance'][task].append(value)
    
    def log_loss(self, task: str, loss: float):
        """Log training loss"""
        if 'task_losses' not in self.metrics:
            self.metrics['task_losses'] = {}
        if task not in self.metrics['task_losses']:
            self.metrics['task_losses'][task] = []
        self.metrics['task_losses'][task].append(loss)
    
    def save(self):
        """Save metrics to file"""
        filepath = os.path.join(self.log_dir, f"training_metrics_{self.timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved training metrics to {filepath}")
    
    def plot_results(self):
        """Plot training results"""
        episodes = [e['episode'] for e in self.metrics['episodes']]
        scores = [e['score'] for e in self.metrics['episodes']]
        
        plt.figure(figsize=(12, 8))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(episodes, scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        
        # Plot task performance
        plt.subplot(2, 2, 2)
        for task, values in self.metrics['task_performance'].items():
            if values:
                plt.plot(episodes[:len(values)], values, label=task)
        plt.title('Task Performance')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True)
        
        # Plot losses
        if 'task_losses' in self.metrics:
            plt.subplot(2, 2, 3)
            for task, losses in self.metrics['task_losses'].items():
                if losses:
                    plt.plot(losses, label=f"{task} loss")
            plt.title('Training Losses')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Plot moving average
        plt.subplot(2, 2, 4)
        window = 50
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(scores)), moving_avg)
            plt.title(f'Score Moving Average (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
            plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f"training_plot_{self.timestamp}.png")
        plt.savefig(plot_path)
        print(f"Saved training plot to {plot_path}")
        plt.close()


def train_task_agent(task_name: str, episodes: int = 500, render: bool = False) -> TaskSpecificAgent:
    """Train a single task-specific agent"""
    print("\n" + "="*60)
    print(f"TRAINING {task_name.upper()} AGENT")
    print("="*60)
    print(f"Episodes: {episodes}")
    print(f"Render: {render}")
    print("="*60 + "\n")
    
    # Initialize game and agent
    game = AsteroidsGame(render=render)
    state = game.reset()
    state_size = len(game.get_state_vector())
    action_size = game.get_action_space()
    
    agent = create_task_agent(task_name, state_size, action_size)
    logger = TrainingLogger(f"training_logs/{task_name}")
    
    # Training metrics
    scores = []
    recent_scores = []
    best_score = -float('inf')
    episode_times = []
    
    print(f"State size: {state_size}")
    print(f"Action space: {action_size}")
    print("\nStarting training...")
    print("-" * 60)
    
    import time
    start_time = time.time()
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        steps = 0
        done = False
        episode_losses = []
        
        while not done and steps < 1000:
            # Get state vector
            state_vector = game.get_state_vector()
            
            # Select action
            action = agent.act(state_vector)
            
            # Take action
            next_state, reward, done, info = game.step(action)
            next_state_vector = game.get_state_vector()
            
            # Apply task-specific reward shaping
            shaped_reward = shape_reward_for_task(task_name, reward, state_vector, 
                                                 next_state_vector, info)
            
            # Store experience
            agent.remember(state_vector, action, shaped_reward, next_state_vector, done)
            
            # Train if enough samples
            if len(agent.memory) > 32:
                loss = agent.train()
                if loss is not None:
                    episode_losses.append(loss)
                    logger.log_loss(task_name, loss)
            
            total_reward += reward
            steps += 1
            
            if render:
                game.render()
        
        # Episode complete
        scores.append(total_reward)
        recent_scores.append(total_reward)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        avg_score = np.mean(recent_scores)
        
        # Log episode
        task_metrics = {
            task_name + '_confidence': agent.get_confidence(state_vector),
            task_name + '_importance': agent.importance_score
        }
        logger.log_episode(episode, total_reward, steps, task_metrics)
        
        # Print progress
        if episode % 5 == 0 or episode == episodes - 1:
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
            eta = (episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
            
            # Progress bar
            progress = (episode + 1) / episodes
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '#' * filled + '-' * (bar_length - filled)
            
            print(f"\r[{bar}] {progress*100:5.1f}% | "
                  f"Ep {episode+1:4d}/{episodes} | "
                  f"Score: {total_reward:6.1f} | "
                  f"Avg: {avg_score:6.1f} | "
                  f"Best: {best_score:6.1f} | "
                  f"ETA: {eta:4.0f}s", end='', flush=True)
            
            if episode % 50 == 0 and episode > 0:
                print()  # New line every 50 episodes
        
        # Save best model
        if avg_score > best_score and episode > 50:
            best_score = avg_score
            model_path = f"models/best_{task_name}_agent.pth"
            agent.save(model_path)
            print(f"  -> New best average score: {best_score:.2f}")
    
    # Save final model and logs
    final_path = f"models/{task_name}_agent_final.pth"
    agent.save(final_path)
    logger.save()
    logger.plot_results()
    
    game.close()
    
    # Print final summary
    print("\n\n" + "="*60)
    print(f"TRAINING COMPLETE: {task_name.upper()} AGENT")
    print("="*60)
    print(f"Total episodes: {episodes}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Final average score (last 100): {np.mean(recent_scores):.2f}")
    print(f"Best average score: {best_score:.2f}")
    print(f"Models saved to: models/")
    print("="*60 + "\n")
    
    return agent


def shape_reward_for_task(task_name: str, base_reward: float, 
                         state: np.ndarray, next_state: np.ndarray, 
                         info: Dict) -> float:
    """Apply task-specific reward shaping"""
    
    if task_name == "avoidance":
        # Reward for staying alive and avoiding threats
        survival_bonus = 0.1
        
        # Penalize getting too close to asteroids
        # Assuming threat distances are in state indices 20-27
        if len(state) > 27:
            min_threat_dist = min(state[20:28])
            if min_threat_dist < 0.2:
                base_reward -= 0.5
            elif min_threat_dist > 0.5:
                base_reward += 0.2
        
        return base_reward + survival_bonus
    
    elif task_name == "combat":
        # Reward for destroying asteroids
        if 'asteroids_destroyed' in info:
            base_reward += info['asteroids_destroyed'] * 5.0
        
        # Small penalty for not shooting when targets are available
        # This encourages aggressive play for combat agent
        if len(state) > 20 and state[20] < 0.3:  # Close target
            base_reward -= 0.1
        
        return base_reward
    
    elif task_name == "navigation":
        # Reward for efficient movement
        # Assuming player velocity is in state
        if len(state) > 4:
            speed = np.sqrt(state[2]**2 + state[3]**2)
            if 0.3 < speed < 0.7:  # Optimal speed range
                base_reward += 0.1
        
        # Reward for exploring (being away from edges)
        if len(state) > 1:
            center_dist = np.sqrt((state[0] - 0.5)**2 + (state[1] - 0.5)**2)
            if center_dist < 0.3:
                base_reward += 0.05
        
        return base_reward
    
    return base_reward


def train_meta_agent(episodes: int = 1000, render: bool = False):
    """Train meta-learning agent with pre-trained task agents"""
    print(f"\nTraining Meta-Learning Agent for {episodes} episodes...")
    
    # Initialize game
    game = AsteroidsGame(render=render)
    state = game.reset()
    state_size = len(game.get_state_vector())
    action_size = game.get_action_space()
    
    # Create meta agent
    agent = MetaLearningAgent(state_size, action_size)
    
    # Load pre-trained task agents if available
    for task_name in ['avoidance', 'combat', 'navigation']:
        model_path = f"models/best_{task_name}_agent.pth"
        if os.path.exists(model_path):
            agent.task_agents[task_name].load(model_path)
            print(f"Loaded pre-trained {task_name} agent")
    
    logger = TrainingLogger("training_logs/meta")
    
    scores = []
    recent_scores = []
    best_score = -float('inf')
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        steps = 0
        done = False
        task_usage = {task: 0 for task in agent.task_agents.keys()}
        
        while not done and steps < 1000:
            state_vector = game.get_state_vector()
            
            # Meta-agent selects action
            action = agent.act(state_vector)
            
            # Track task usage
            if agent.current_task:
                task_usage[agent.current_task] += 1
            
            # Take action
            next_state, reward, done, info = game.step(action)
            next_state_vector = game.get_state_vector()
            
            # Store experience
            agent.remember(state_vector, action, reward, next_state_vector, done)
            
            # Train periodically
            if steps % 4 == 0:
                loss = agent.train()
            
            total_reward += reward
            steps += 1
            
            if render:
                game.render()
        
        # Episode complete
        scores.append(total_reward)
        recent_scores.append(total_reward)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        avg_score = np.mean(recent_scores)
        
        # Calculate task usage percentages
        total_usage = sum(task_usage.values())
        task_metrics = {}
        for task, count in task_usage.items():
            usage_pct = (count / total_usage * 100) if total_usage > 0 else 0
            task_metrics[f"{task}_usage"] = usage_pct
        
        logger.log_episode(episode, total_reward, steps, task_metrics)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Score: {total_reward:6.1f} | "
                  f"Avg Score: {avg_score:6.1f}")
            print(f"  Task usage: {', '.join([f'{k}: {v:.1f}%' for k, v in task_metrics.items()])}")
        
        # Save best model
        if avg_score > best_score and episode > 50:
            best_score = avg_score
            model_path = "models/best_meta_agent.pth"
            agent.save(model_path)
            print(f"  -> New best average score: {best_score:.2f}")
    
    # Save final model
    agent.save("models/meta_agent_final.pth")
    logger.save()
    logger.plot_results()
    
    game.close()
    
    print(f"\nMeta-agent training complete")
    print(f"Final average score: {np.mean(recent_scores):.2f}")
    
    return agent


def main():
    """Main training routine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI agents')
    parser.add_argument('--task', type=str, choices=['all', 'avoidance', 'combat', 
                       'navigation', 'meta'], default='all',
                       help='Which agent(s) to train')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render game during training')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    
    if args.task == 'all':
        # Train all task agents first
        for task in ['avoidance', 'combat', 'navigation']:
            train_task_agent(task, args.episodes, args.render)
        
        # Then train meta agent
        train_meta_agent(args.episodes, args.render)
    
    elif args.task == 'meta':
        train_meta_agent(args.episodes, args.render)
    
    else:
        train_task_agent(args.task, args.episodes, args.render)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()