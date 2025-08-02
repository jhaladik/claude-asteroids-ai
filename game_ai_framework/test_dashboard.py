"""
Simplified dashboard to test game rendering
"""

import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
import numpy as np

def test_simple_dashboard():
    """Test basic dashboard functionality"""
    pygame.init()
    
    # Create main window
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Test Dashboard")
    clock = pygame.time.Clock()
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (50, 50, 50)
    GREEN = (0, 255, 0)
    
    # Create game
    print("Creating Asteroids game...")
    game = AsteroidsGame(render=True)
    game.reset()
    
    # Create agent
    state_size = len(game.get_state_vector())
    action_size = game.get_action_space()
    agent = MetaLearningAgent(state_size, action_size)
    
    # Try to load models
    for task in ['avoidance', 'combat', 'navigation']:
        path = f"models/best_{task}_agent.pth"
        if os.path.exists(path):
            try:
                agent.task_agents[task].load(path)
                print(f"Loaded {task} model")
            except:
                pass
    
    # Game state
    score = 0
    episode = 0
    steps = 0
    
    # Main loop
    running = True
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    print("Starting main loop...")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Reset game
                    game.reset()
                    score = 0
                    steps = 0
                    episode += 1
        
        # Clear screen
        screen.fill(BLACK)
        
        # Game step
        state_vector = game.get_state_vector()
        action = agent.act(state_vector, training=False)
        next_state, reward, done, info = game.step(action)
        score += reward
        steps += 1
        
        # Render game
        game.render()
        
        # Draw game area
        game_area = pygame.Rect(50, 50, 800, 600)
        pygame.draw.rect(screen, GRAY, game_area, 2)
        
        # Try to get game surface
        if hasattr(game, 'screen') and game.screen:
            try:
                # Scale and blit game surface
                game_surface = game.screen
                scaled = pygame.transform.scale(game_surface, (800, 600))
                screen.blit(scaled, (50, 50))
            except Exception as e:
                error_text = small_font.render(f"Render error: {str(e)}", True, (255, 0, 0))
                screen.blit(error_text, (60, 60))
        
        # Draw stats
        stats_x = 900
        stats_y = 50
        
        title = font.render("Game Stats", True, WHITE)
        screen.blit(title, (stats_x, stats_y))
        
        stats = [
            f"Episode: {episode}",
            f"Steps: {steps}",
            f"Score: {score:.1f}",
            f"Current Task: {getattr(agent, 'current_task', 'N/A')}",
            "",
            "Controls:",
            "SPACE - Reset",
            "ESC - Exit"
        ]
        
        y = stats_y + 50
        for stat in stats:
            text = small_font.render(stat, True, WHITE)
            screen.blit(text, (stats_x, y))
            y += 30
        
        # Reset if done
        if done:
            game.reset()
            score = 0
            steps = 0
            episode += 1
        
        # Update display
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    # Cleanup
    game.close()
    pygame.quit()

if __name__ == "__main__":
    print("Testing simplified dashboard...")
    try:
        test_simple_dashboard()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()