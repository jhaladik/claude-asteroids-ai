"""
Main entry point for the Game AI Framework
Demonstrates training agents on games with the modular system
"""

import sys
import os
import pygame
import numpy as np
from typing import Dict, Any

# Add framework to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
from agents.task_agents import create_task_agent
from core.base_game import TaskContext


def demonstrate_framework(game_name="asteroids"):
    """Demonstrate the modular framework"""
    print("Game AI Framework - Modular Meta-Learning System")
    print("=" * 60)
    
    # Initialize game
    if game_name.lower() == "snake":
        game = SnakeGame(render=True)
    else:
        game = AsteroidsGame(render=True)
    state = game.reset()
    
    # Get state vector size
    state_vector = game.get_state_vector()
    state_size = len(state_vector)
    action_size = game.get_action_space()
    
    print(f"\nGame: {game.__class__.__name__}")
    print(f"State size: {state_size}")
    print(f"Action space: {action_size}")
    print(f"Actions: {game.get_action_meanings()}")
    
    # Analyze initial task context
    context = game.analyze_task_context()
    print(f"\nInitial Task Context:")
    print(f"  Avoidance priority: {context.avoidance_priority:.2f}")
    print(f"  Combat priority: {context.combat_priority:.2f}")
    print(f"  Navigation priority: {context.navigation_priority:.2f}")
    print(f"  Survival priority: {context.survival_priority:.2f}")
    print(f"  Active tasks: {context.get_active_tasks()}")
    
    # Create meta-learning agent
    print("\nCreating Meta-Learning Agent...")
    agent = MetaLearningAgent(state_size, action_size)
    
    # Demonstrate task selection
    print("\nTask Agent Confidence Scores:")
    for task_name, task_agent in agent.task_agents.items():
        confidence = task_agent.get_confidence(state_vector)
        print(f"  {task_name}: {confidence:.2f}")
    
    # Game loop
    print("\n\nStarting demonstration...")
    print("Press ESC to exit, SPACE to pause")
    print("-" * 60)
    
    running = True
    paused = False
    step_count = 0
    total_reward = 0
    
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
        
        if not paused:
            # Get current state
            state_vector = game.get_state_vector()
            
            # Agent selects action
            action = agent.act(state_vector)
            
            # Execute action
            next_state, reward, done, info = game.step(action)
            next_state_vector = game.get_state_vector()
            
            # Store experience
            agent.remember(state_vector, action, reward, next_state_vector, done)
            
            # Update metrics
            total_reward += reward
            step_count += 1
            
            # Train periodically
            if step_count % 100 == 0:
                loss = agent.train()
                if loss is not None:
                    print(f"Step {step_count}: Loss = {loss:.4f}, Total Reward = {total_reward:.2f}")
            
            # Render
            game.render()
            
            # Show current task and metrics
            if step_count % 30 == 0:  # Every 0.5 seconds at 60 FPS
                context = game.analyze_task_context()
                semantic = game.get_semantic_features()
                metrics = agent.get_metrics()
                
                print(f"\nStep {step_count}:")
                print(f"  Current task: {agent.current_task}")
                print(f"  Threat level: {semantic['threat_level']:.2f}")
                print(f"  Task usage: {[f'{k}: {v:.2%}' for k, v in metrics.items() if 'usage_rate' in k]}")
            
            if done:
                score_key = 'score' if 'score' in info else 'lives'
                print(f"\nGame Over! Score: {info[score_key]}")
                state = game.reset()
                total_reward = 0
        
        else:
            # Just render when paused
            game.render()
        
        clock.tick(60)  # 60 FPS
    
    # Clean up
    game.close()
    
    # Show final metrics
    print("\n\nFinal Metrics:")
    metrics = agent.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Demonstrate transfer potential
    print("\n\nTransfer Potential Analysis:")
    # Create a hypothetical new game state
    new_state = np.random.randn(state_size)
    transfer_scores = agent.analyze_transfer_potential(new_state)
    for task, score in transfer_scores.items():
        print(f"  {task}: {score:.2f}")


def train_individual_tasks():
    """Train individual task agents separately"""
    print("\nTraining Individual Task Agents")
    print("=" * 60)
    
    # Initialize game
    game = AsteroidsGame(render=False)
    state = game.reset()
    state_size = len(game.get_state_vector())
    action_size = game.get_action_space()
    
    # Train each task agent
    tasks = ['avoidance', 'combat', 'navigation']
    
    for task_name in tasks:
        print(f"\nTraining {task_name} agent...")
        agent = create_task_agent(task_name, state_size, action_size)
        
        # Simple training loop
        for episode in range(100):
            state = game.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_vector = game.get_state_vector()
                action = agent.act(state_vector)
                next_state, reward, done, info = game.step(action)
                next_state_vector = game.get_state_vector()
                
                # Task-specific reward shaping could go here
                agent.remember(state_vector, action, reward, next_state_vector, done)
                total_reward += reward
                
                # Train
                if len(agent.memory) > 32:
                    agent.train()
            
            if episode % 20 == 0:
                print(f"  Episode {episode}: Total Reward = {total_reward:.2f}")
        
        # Save trained agent
        agent.save(f"models/{task_name}_agent.pth")
        print(f"  Saved {task_name} agent")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train_tasks":
            # Create models directory
            os.makedirs("models", exist_ok=True)
            train_individual_tasks()
        elif sys.argv[1] == "snake":
            demonstrate_framework("snake")
        elif sys.argv[1] == "asteroids":
            demonstrate_framework("asteroids")
        else:
            print("Usage: python main.py [asteroids|snake|train_tasks]")
    else:
        # Show menu
        print("\nGame AI Framework")
        print("1. Asteroids Demo")
        print("2. Snake Demo")
        print("3. Train Task Agents")
        choice = input("\nSelect option (1-3): ")
        
        if choice == "1":
            demonstrate_framework("asteroids")
        elif choice == "2":
            demonstrate_framework("snake")
        elif choice == "3":
            os.makedirs("models", exist_ok=True)
            train_individual_tasks()
        else:
            print("Invalid choice")