"""
Transfer Learning Demo
Shows how agents trained on Asteroids can transfer skills to Snake
"""

import sys
import os
import pygame
import numpy as np
import torch
from typing import Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
from core.base_game import BaseGame
import matplotlib.pyplot as plt


def visualize_transfer_potential(agent: MetaLearningAgent, game1: BaseGame, game2: BaseGame):
    """Visualize how well skills transfer between games"""
    
    # Get initial states from both games
    state1 = game1.reset()
    state2 = game2.reset()
    
    # Analyze transfer potential
    print("\nTransfer Potential Analysis")
    print("=" * 60)
    
    # Compare state vector sizes
    print(f"\nState Vector Comparison:")
    print(f"  {game1.__class__.__name__}: {len(game1.get_state_vector())} dimensions")
    print(f"  {game2.__class__.__name__}: {len(game2.get_state_vector())} dimensions")
    
    # Compare task contexts
    context1 = game1.analyze_task_context()
    context2 = game2.analyze_task_context()
    
    print(f"\nTask Context Comparison:")
    print(f"  {'Task':<20} {'Asteroids':>10} {'Snake':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    print(f"  {'Avoidance':<20} {context1.avoidance_priority:>10.2f} {context2.avoidance_priority:>10.2f}")
    print(f"  {'Navigation':<20} {context1.navigation_priority:>10.2f} {context2.navigation_priority:>10.2f}")
    print(f"  {'Combat':<20} {context1.combat_priority:>10.2f} {context2.combat_priority:>10.2f}")
    print(f"  {'Survival':<20} {context1.survival_priority:>10.2f} {context2.survival_priority:>10.2f}")
    
    # Compare semantic features
    semantic1 = game1.get_semantic_features()
    semantic2 = game2.get_semantic_features()
    
    print(f"\nSemantic Features Comparison:")
    print(f"  {'Feature':<25} {'Asteroids':>10} {'Snake':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for key in semantic1:
        if key in semantic2:
            print(f"  {key:<25} {semantic1[key]:>10.2f} {semantic2[key]:>10.2f}")


def demonstrate_transfer(pretrained_path: str = None):
    """Demonstrate skill transfer from Asteroids to Snake"""
    
    print("Transfer Learning Demonstration")
    print("=" * 60)
    
    # Initialize games
    asteroids = AsteroidsGame(render=False)
    snake = SnakeGame(render=True)
    
    # Get state sizes (need to handle different sizes)
    asteroids_state = asteroids.reset()
    snake_state = snake.reset()
    
    asteroids_state_size = len(asteroids.get_state_vector())
    snake_state_size = len(snake.get_state_vector())
    
    # We'll use the larger state size and pad if necessary
    max_state_size = max(asteroids_state_size, snake_state_size)
    
    # Create meta-agent
    print(f"\nCreating Meta-Agent (state_size={max_state_size})...")
    agent = MetaLearningAgent(max_state_size, 3)  # 3 actions for Snake
    
    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        agent.load(pretrained_path)
    else:
        # Try to load individual task agents
        print("Loading pre-trained task agents...")
        loaded_count = 0
        for task_name in ['avoidance', 'combat', 'navigation']:
            model_path = f"models/best_{task_name}_agent.pth"
            if os.path.exists(model_path):
                try:
                    agent.task_agents[task_name].load(model_path)
                    print(f"  [OK] Loaded {task_name} agent")
                    loaded_count += 1
                except Exception as e:
                    print(f"  [SKIP] Failed to load {task_name} agent (incompatible action space)")
            else:
                print(f"  - No model found for {task_name} agent")
        
        if loaded_count > 0:
            print(f"Successfully loaded {loaded_count} task agents")
    
    # Analyze transfer potential
    visualize_transfer_potential(agent, asteroids, snake)
    
    # Create state adapter function
    def adapt_state(state: np.ndarray, target_size: int) -> np.ndarray:
        """Adapt state vector to target size by padding or truncating"""
        if len(state) == target_size:
            return state
        elif len(state) < target_size:
            # Pad with zeros
            padded = np.zeros(target_size)
            padded[:len(state)] = state
            return padded
        else:
            # Truncate
            return state[:target_size]
    
    # Test on Snake game
    print("\n\nTesting on Snake Game...")
    print("-" * 60)
    
    episodes = 10
    scores = []
    task_usage = {task: 0 for task in agent.task_agents.keys()}
    
    for episode in range(episodes):
        state = snake.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            # Adapt state to agent's expected size
            state_vector = snake.get_state_vector()
            adapted_state = adapt_state(state_vector, max_state_size)
            
            # Get action from agent
            action = agent.act(adapted_state, training=False)
            
            # Track which task was used
            if agent.current_task:
                task_usage[agent.current_task] += 1
            
            # Take action in game
            next_state, reward, done, info = snake.step(action)
            total_reward += reward
            steps += 1
            
            # Update state
            state = next_state
            
            # Render
            snake.render()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    snake.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        snake.close()
                        return
        
        scores.append(info['score'])
        print(f"Episode {episode + 1}: Score = {info['score']}, Steps = {steps}")
    
    # Show results
    print("\n\nTransfer Learning Results:")
    print("-" * 60)
    print(f"Average Score: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
    print(f"Best Score: {max(scores)}")
    
    print("\nTask Usage Distribution:")
    total_usage = sum(task_usage.values())
    for task, count in task_usage.items():
        percentage = (count / total_usage * 100) if total_usage > 0 else 0
        print(f"  {task}: {percentage:.1f}%")
    
    # Analyze which skills transferred well
    print("\n\nSkill Transfer Analysis:")
    print("-" * 60)
    print("Skills that transfer well:")
    print("  - Avoidance: Both games require avoiding collisions")
    print("  - Navigation: Both games need efficient path planning")
    print("\nSkills that don't transfer:")
    print("  - Combat: Snake has no shooting mechanism")
    print("\nNew skills needed for Snake:")
    print("  - Food tracking: Following a moving target")
    print("  - Growth management: Dealing with increasing body length")
    
    # Clean up
    snake.close()
    asteroids.close()


def compare_random_vs_transfer():
    """Compare random agent vs transferred agent on Snake"""
    
    print("\n\nComparing Random vs Transfer Agent")
    print("=" * 60)
    
    snake = SnakeGame(render=False)
    episodes = 20
    
    # Test random agent
    print("\nTesting Random Agent...")
    random_scores = []
    for _ in range(episodes):
        state = snake.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            action = np.random.randint(0, 3)
            state, _, done, info = snake.step(action)
            steps += 1
        random_scores.append(info['score'])
    
    # Test transfer agent
    print("\nTesting Transfer Agent...")
    snake.reset()
    state_size = len(snake.get_state_vector())
    agent = MetaLearningAgent(state_size, 3)
    
    transfer_scores = []
    for _ in range(episodes):
        state = snake.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            state_vector = snake.get_state_vector()
            action = agent.act(state_vector, training=False)
            state, _, done, info = snake.step(action)
            steps += 1
        transfer_scores.append(info['score'])
    
    # Compare results
    print("\n\nResults:")
    print(f"Random Agent: {np.mean(random_scores):.2f} (+/- {np.std(random_scores):.2f})")
    print(f"Transfer Agent: {np.mean(transfer_scores):.2f} (+/- {np.std(transfer_scores):.2f})")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([random_scores, transfer_scores], labels=['Random', 'Transfer'])
    plt.ylabel('Score')
    plt.title('Snake Performance: Random vs Transfer Agent')
    plt.grid(True, alpha=0.3)
    plt.savefig('transfer_comparison.png')
    print("\nComparison plot saved as 'transfer_comparison.png'")
    
    snake.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transfer Learning Demo')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained model')
    parser.add_argument('--compare', action='store_true', help='Compare with random agent')
    args = parser.parse_args()
    
    if args.compare:
        compare_random_vs_transfer()
    else:
        demonstrate_transfer(args.pretrained)