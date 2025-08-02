"""
Test script to identify bugs in game playing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
import numpy as np

def test_asteroids():
    """Test Asteroids game"""
    print("\n" + "="*60)
    print("TESTING ASTEROIDS GAME")
    print("="*60)
    
    try:
        # Create game
        game = AsteroidsGame(render=False)
        print("[OK] Game created")
        
        # Reset game
        state = game.reset()
        print("[OK] Game reset")
        
        # Get state vector
        state_vector = game.get_state_vector()
        print(f"[OK] State vector size: {len(state_vector)}")
        
        # Get action space
        action_space = game.get_action_space()
        print(f"[OK] Action space: {action_space}")
        
        # Test a few steps
        for i in range(10):
            action = np.random.randint(0, action_space)
            next_state, reward, done, info = game.step(action)
            print(f"[OK] Step {i+1}: reward={reward:.2f}, done={done}")
            
            if done:
                state = game.reset()
                print("[OK] Game reset after done")
        
        # Test state methods
        task_context = game.analyze_task_context()
        print(f"[OK] Task context: avoidance={task_context.avoidance_priority:.2f}")
        
        semantic_features = game.get_semantic_features()
        print(f"[OK] Semantic features: threat_level={semantic_features['threat_level']:.2f}")
        
        game.close()
        print("[OK] Game closed")
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_snake():
    """Test Snake game"""
    print("\n" + "="*60)
    print("TESTING SNAKE GAME")
    print("="*60)
    
    try:
        # Create game
        game = SnakeGame(render=False)
        print("[OK] Game created")
        
        # Reset game
        state = game.reset()
        print("[OK] Game reset")
        
        # Get state vector
        state_vector = game.get_state_vector()
        print(f"[OK] State vector size: {len(state_vector)}")
        
        # Get action space
        action_space = game.get_action_space()
        print(f"[OK] Action space: {action_space}")
        
        # Test a few steps
        for i in range(10):
            action = np.random.randint(0, action_space)
            next_state, reward, done, info = game.step(action)
            print(f"[OK] Step {i+1}: reward={reward:.2f}, done={done}, score={info['score']}")
            
            if done:
                state = game.reset()
                print("[OK] Game reset after done")
        
        # Test state methods
        task_context = game.analyze_task_context()
        print(f"[OK] Task context: avoidance={task_context.avoidance_priority:.2f}")
        
        semantic_features = game.get_semantic_features()
        print(f"[OK] Semantic features: threat_level={semantic_features['threat_level']:.2f}")
        
        game.close()
        print("[OK] Game closed")
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_agents():
    """Test agent creation and loading"""
    print("\n" + "="*60)
    print("TESTING AGENTS")
    print("="*60)
    
    try:
        # Test with Asteroids state size
        state_size = 38
        action_size = 5
        
        # Create meta agent
        agent = MetaLearningAgent(state_size, action_size)
        print("[OK] Meta agent created")
        
        # Test action selection
        state = np.random.randn(state_size)
        action = agent.act(state, training=False)
        print(f"[OK] Action selected: {action}")
        
        # Check task agents
        for task_name, task_agent in agent.task_agents.items():
            confidence = task_agent.get_confidence(state)
            print(f"[OK] {task_name} confidence: {confidence:.2f}")
        
        # Try loading models
        loaded = 0
        for task in ['avoidance', 'combat', 'navigation']:
            path = f"models/best_{task}_agent.pth"
            if os.path.exists(path):
                try:
                    agent.task_agents[task].load(path)
                    loaded += 1
                    print(f"[OK] Loaded {task} model")
                except Exception as e:
                    print(f"[WARNING] Could not load {task}: {e}")
        
        print(f"[INFO] Loaded {loaded} models")
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_game_agent_integration():
    """Test game and agent integration"""
    print("\n" + "="*60)
    print("TESTING GAME-AGENT INTEGRATION")
    print("="*60)
    
    try:
        # Test Asteroids with agent
        game = AsteroidsGame(render=False)
        state = game.reset()
        state_size = len(game.get_state_vector())
        action_size = game.get_action_space()
        
        agent = MetaLearningAgent(state_size, action_size)
        print("[OK] Asteroids + Meta agent initialized")
        
        # Run a few steps
        for i in range(20):
            state_vector = game.get_state_vector()
            action = agent.act(state_vector, training=False)
            next_state, reward, done, info = game.step(action)
            
            if done:
                state = game.reset()
                print(f"[OK] Episode complete at step {i+1}")
        
        game.close()
        
        # Test Snake with different action space
        game = SnakeGame(render=False)
        state = game.reset()
        state_size = len(game.get_state_vector())
        action_size = game.get_action_space()
        
        # Need new agent for different action space
        agent = MetaLearningAgent(state_size, action_size)
        print("[OK] Snake + Meta agent initialized")
        
        # Run a few steps
        for i in range(20):
            state_vector = game.get_state_vector()
            action = agent.act(state_vector, training=False)
            next_state, reward, done, info = game.step(action)
            
            if done:
                state = game.reset()
                print(f"[OK] Episode complete at step {i+1}, score: {info['score']}")
        
        game.close()
        print("[OK] Integration tests passed")
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Running comprehensive bug tests...")
    
    test_asteroids()
    test_snake()
    test_agents()
    test_game_agent_integration()
    
    print("\n" + "="*60)
    print("BUG TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()