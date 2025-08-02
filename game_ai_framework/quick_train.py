"""
Quick training script for demonstration
Trains all agents with fewer episodes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_agents import train_task_agent, train_meta_agent

def main():
    print("\n" + "="*60)
    print("QUICK TRAINING DEMO")
    print("Training all agents with 50 episodes each")
    print("="*60)
    
    # Quick train each task agent
    episodes = 50
    
    print("\nPhase 1: Training Task-Specific Agents")
    print("-"*60)
    
    for task in ['avoidance', 'combat', 'navigation']:
        agent = train_task_agent(task, episodes=episodes, render=False)
    
    print("\nPhase 2: Training Meta-Learning Agent")
    print("-"*60)
    
    meta_agent = train_meta_agent(episodes=episodes, render=False)
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("Models saved to: game_ai_framework/models/")
    print("Ready for transfer learning demo")
    print("="*60)

if __name__ == "__main__":
    main()