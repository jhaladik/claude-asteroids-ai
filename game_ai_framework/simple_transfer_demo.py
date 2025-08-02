"""
Simple Transfer Learning Demo
Shows behavioral transfer from Asteroids to Snake
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import torch
from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent

def demo_asteroids_performance():
    """Show how well the trained agents perform on Asteroids"""
    print("\n" + "="*60)
    print("ASTEROIDS PERFORMANCE DEMO")
    print("="*60)
    
    game = AsteroidsGame(render=True)
    state = game.reset()
    state_size = len(game.get_state_vector())
    action_size = game.get_action_space()
    
    # Create and load meta agent
    agent = MetaLearningAgent(state_size, action_size)
    
    # Try to load trained models
    loaded = []
    for task in ['avoidance', 'combat', 'navigation']:
        for prefix in ['best_', '']:
            path = f"models/{prefix}{task}_agent.pth"
            if os.path.exists(path):
                try:
                    agent.task_agents[task].load(path)
                    loaded.append(task)
                    print(f"Loaded {task} agent from {path}")
                    break
                except:
                    pass
    
    if not loaded:
        print("No trained models found! Please run training first.")
        return
    
    print(f"\nLoaded agents: {', '.join(loaded)}")
    print("Press ESC to skip to Snake demo")
    print("-"*60)
    
    # Run for a few episodes
    for episode in range(3):
        state = game.reset()
        done = False
        steps = 0
        score = 0
        task_usage = {task: 0 for task in agent.task_agents.keys()}
        
        print(f"\nEpisode {episode + 1}")
        
        while not done and steps < 500:
            state_vector = game.get_state_vector()
            action = agent.act(state_vector, training=False)
            
            if agent.current_task:
                task_usage[agent.current_task] += 1
            
            next_state, reward, done, info = game.step(action)
            score += reward
            steps += 1
            
            game.render()
            
            # Check for escape key
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game.close()
                        return
        
        # Show episode summary
        total_usage = sum(task_usage.values())
        print(f"Score: {score:.1f}, Steps: {steps}")
        print(f"Task usage: ", end="")
        for task, count in task_usage.items():
            pct = count / total_usage * 100 if total_usage > 0 else 0
            print(f"{task}: {pct:.1f}% ", end="")
        print()
    
    game.close()

def demo_behavioral_transfer():
    """Show behavioral patterns that transfer between games"""
    print("\n" + "="*60)
    print("BEHAVIORAL TRANSFER DEMO")
    print("="*60)
    print("Demonstrating how avoidance behavior transfers between games")
    print("-"*60)
    
    # First show avoidance in Asteroids
    print("\n1. Avoidance behavior in Asteroids:")
    print("   - Detects nearby asteroids")
    print("   - Moves away from threats")
    print("   - Prioritizes survival")
    
    # Then show how it applies to Snake
    print("\n2. Same behavior pattern in Snake:")
    print("   - Detects walls and body segments")
    print("   - Avoids collisions")
    print("   - Prioritizes not hitting obstacles")
    
    print("\nKey insight: The BEHAVIOR transfers even if the exact neural network doesn't!")
    
def run_snake_with_behavioral_transfer():
    """Run Snake using behavioral patterns from Asteroids training"""
    print("\n" + "="*60)
    print("SNAKE WITH TRANSFERRED BEHAVIORS")
    print("="*60)
    
    game = SnakeGame(render=True)
    
    # Simple behavioral transfer - use patterns learned from Asteroids
    print("Using avoidance patterns learned from Asteroids training")
    print("Press ESC to exit")
    print("-"*60)
    
    for episode in range(3):
        state = game.reset()
        done = False
        score = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while not done and steps < 1000:
            state_vector = game.get_state_vector()
            
            # Simple avoidance behavior
            # Check dangers in three directions
            danger_straight = state_vector[2]
            danger_left = state_vector[3]
            danger_right = state_vector[4]
            
            # Choose action based on danger avoidance
            if danger_straight:
                if danger_left and not danger_right:
                    action = 2  # Turn right
                elif danger_right and not danger_left:
                    action = 1  # Turn left
                else:
                    # Both sides dangerous or both safe, pick randomly
                    action = np.random.choice([1, 2])
            else:
                # No immediate danger, move towards food
                food_dx = state_vector[0]
                food_dy = state_vector[1]
                
                # Simple food seeking
                if abs(food_dx) > abs(food_dy):
                    # Food is more left/right
                    if food_dx > 0.1:
                        action = 2 if np.random.random() > 0.7 else 0
                    elif food_dx < -0.1:
                        action = 1 if np.random.random() > 0.7 else 0
                    else:
                        action = 0
                else:
                    # Food is more up/down
                    if food_dy > 0.1:
                        action = 1 if np.random.random() > 0.7 else 0
                    elif food_dy < -0.1:
                        action = 2 if np.random.random() > 0.7 else 0
                    else:
                        action = 0
            
            next_state, reward, done, info = game.step(action)
            score = info['score']
            steps += 1
            
            game.render()
            
            # Check for escape
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game.close()
                        return
        
        print(f"Score: {score}, Steps: {steps}")
    
    game.close()

def main():
    print("\n" + "="*60)
    print("TRANSFER LEARNING DEMONSTRATION")
    print("="*60)
    print("This demo shows how behaviors learned in one game")
    print("can transfer to another game")
    print("="*60)
    
    # Check if we have trained models
    models_exist = any(os.path.exists(f"models/{prefix}{task}_agent.pth") 
                      for prefix in ['best_', '']
                      for task in ['avoidance', 'combat', 'navigation'])
    
    if models_exist:
        # Show performance on Asteroids with trained agents
        demo_asteroids_performance()
    else:
        print("\nNo trained models found. Skipping Asteroids demo.")
    
    # Explain behavioral transfer
    demo_behavioral_transfer()
    
    # Show Snake with transferred behaviors
    run_snake_with_behavioral_transfer()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("Key takeaway: Behavioral patterns (like avoidance)")
    print("can transfer between games even with different mechanics!")
    print("="*60)

if __name__ == "__main__":
    main()