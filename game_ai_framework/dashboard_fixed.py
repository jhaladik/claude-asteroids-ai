"""
Fixed Dashboard without threading issues
Uses pygame for all visualizations instead of matplotlib
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import pygame_gui
import numpy as np
import json
from datetime import datetime
import threading
from typing import Dict, List, Optional
import math

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
from agents.task_agents import create_task_agent


class DashboardFixed:
    """Fixed dashboard interface without matplotlib threading issues"""
    
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Game Framework Dashboard")
        
        # UI Manager
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        # Layout
        self.sidebar_width = 300
        self.game_area = pygame.Rect(self.sidebar_width, 50, 800, 600)
        self.stats_area = pygame.Rect(self.sidebar_width, 660, 800, 240)
        self.sidebar_area = pygame.Rect(0, 0, self.sidebar_width, height)
        
        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.SIDEBAR_COLOR = (40, 40, 40)
        self.PANEL_COLOR = (50, 50, 50)
        self.TEXT_COLOR = (200, 200, 200)
        self.ACCENT_COLOR = (100, 200, 100)
        self.CHART_COLOR = (100, 200, 100)
        self.TASK_COLORS = {
            'avoidance': (255, 107, 107),
            'combat': (78, 205, 196),
            'navigation': (69, 183, 209)
        }
        
        # Game state
        self.current_game = None
        self.current_agent = None
        self.game_thread = None
        self.running = False
        self.paused = False
        
        # Metrics
        self.episode_scores = []
        self.current_metrics = {}
        self.performance_history = []
        self.max_history = 100
        
        # Create UI elements
        self._create_ui()
        
        # Clock
        self.clock = pygame.time.Clock()
        
    def _create_ui(self):
        """Create UI elements"""
        # Title
        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 10, 280, 40),
            text="AI Game Framework",
            manager=self.ui_manager
        )
        
        # Game Selection
        self.game_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 60, 280, 30),
            text="Select Game:",
            manager=self.ui_manager
        )
        
        self.game_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["Asteroids", "Snake"],
            starting_option="Asteroids",
            relative_rect=pygame.Rect(10, 90, 280, 30),
            manager=self.ui_manager
        )
        
        # Agent Selection
        self.agent_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 130, 280, 30),
            text="Select Agent:",
            manager=self.ui_manager
        )
        
        self.agent_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["Human", "Random", "Avoidance", "Combat", "Navigation", "Meta-Learning"],
            starting_option="Meta-Learning",
            relative_rect=pygame.Rect(10, 160, 280, 30),
            manager=self.ui_manager
        )
        
        # Control Buttons
        self.play_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, 210, 135, 40),
            text="Play",
            manager=self.ui_manager
        )
        
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(155, 210, 135, 40),
            text="Pause",
            manager=self.ui_manager
        )
        
        # Speed control
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 270, 100, 30),
            text="Game Speed:",
            manager=self.ui_manager
        )
        
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(120, 270, 170, 30),
            start_value=1.0,
            value_range=(0.1, 5.0),
            manager=self.ui_manager
        )
        
        # Performance Stats
        self.stats_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 320, 280, 30),
            text="Performance Stats:",
            manager=self.ui_manager
        )
        
        self.stats_text = pygame_gui.elements.UITextBox(
            html_text="No game running",
            relative_rect=pygame.Rect(10, 350, 280, 300),
            manager=self.ui_manager
        )
        
    def _draw_background(self):
        """Draw background panels"""
        self.screen.fill(self.BG_COLOR)
        
        # Sidebar
        pygame.draw.rect(self.screen, self.SIDEBAR_COLOR, self.sidebar_area)
        
        # Game panel
        pygame.draw.rect(self.screen, self.PANEL_COLOR, self.game_area, 2)
        
        # Stats panel
        pygame.draw.rect(self.screen, self.PANEL_COLOR, self.stats_area, 2)
        
        # Headers
        font = pygame.font.Font(None, 24)
        game_text = font.render("Game View", True, self.TEXT_COLOR)
        self.screen.blit(game_text, (self.game_area.x + 10, self.game_area.y - 30))
        
        stats_text = font.render("Real-time Metrics", True, self.TEXT_COLOR)
        self.screen.blit(stats_text, (self.stats_area.x + 10, self.stats_area.y - 30))
        
    def _draw_game(self):
        """Draw current game"""
        if self.current_game:
            # Create a surface for the game
            game_surface = pygame.Surface((self.game_area.width - 4, self.game_area.height - 4))
            game_surface.fill((0, 0, 0))
            
            # Draw game-specific visuals
            if isinstance(self.current_game, AsteroidsGame):
                self._draw_asteroids_game(game_surface)
            elif isinstance(self.current_game, SnakeGame):
                self._draw_snake_game(game_surface)
            
            # Blit to main screen
            self.screen.blit(game_surface, (self.game_area.x + 2, self.game_area.y + 2))
            
            # Draw game info overlay
            font = pygame.font.Font(None, 20)
            game_name = self.current_game.__class__.__name__.replace("Game", "")
            score = self.current_metrics.get('score', 0)
            info_text = font.render(f"{game_name} - Score: {score:.1f}", True, self.TEXT_COLOR)
            self.screen.blit(info_text, (self.game_area.x + 10, self.game_area.y + 10))
        else:
            # Draw placeholder
            font = pygame.font.Font(None, 36)
            text = font.render("No Game Running", True, self.TEXT_COLOR)
            text_rect = text.get_rect(center=self.game_area.center)
            self.screen.blit(text, text_rect)
    
    def _draw_asteroids_game(self, surface):
        """Draw Asteroids game visualization"""
        # Get game state if available
        if hasattr(self.current_game, 'ship') and self.current_game.ship:
            # Draw ship
            ship_x = int(self.current_game.ship.x * surface.get_width() / self.current_game.width)
            ship_y = int(self.current_game.ship.y * surface.get_height() / self.current_game.height)
            pygame.draw.circle(surface, (100, 200, 100), (ship_x, ship_y), 10)
            
            # Draw ship direction
            angle = math.radians(self.current_game.ship.angle)
            end_x = ship_x + int(15 * math.cos(angle))
            end_y = ship_y + int(15 * math.sin(angle))
            pygame.draw.line(surface, (150, 250, 150), (ship_x, ship_y), (end_x, end_y), 2)
        
        # Draw asteroids
        if hasattr(self.current_game, 'asteroids'):
            for asteroid in self.current_game.asteroids[:10]:  # Limit to 10 for performance
                x = int(asteroid.x * surface.get_width() / self.current_game.width)
                y = int(asteroid.y * surface.get_height() / self.current_game.height)
                radius = int(asteroid.radius * surface.get_width() / self.current_game.width)
                pygame.draw.circle(surface, (150, 150, 150), (x, y), radius, 2)
        
        # Draw bullets
        if hasattr(self.current_game, 'bullets'):
            for bullet in self.current_game.bullets:
                x = int(bullet.x * surface.get_width() / self.current_game.width)
                y = int(bullet.y * surface.get_height() / self.current_game.height)
                pygame.draw.circle(surface, (255, 255, 0), (x, y), 3)
    
    def _draw_snake_game(self, surface):
        """Draw Snake game visualization"""
        if not hasattr(self.current_game, 'snake'):
            return
            
        # Calculate cell size
        grid_size = self.current_game.grid_size
        cell_width = surface.get_width() // grid_size
        cell_height = surface.get_height() // grid_size
        
        # Draw grid
        for x in range(grid_size):
            pygame.draw.line(surface, (40, 40, 40), 
                           (x * cell_width, 0), (x * cell_width, surface.get_height()))
        for y in range(grid_size):
            pygame.draw.line(surface, (40, 40, 40), 
                           (0, y * cell_height), (surface.get_width(), y * cell_height))
        
        # Draw snake
        if self.current_game.snake:
            for i, segment in enumerate(self.current_game.snake):
                x = segment[0] * cell_width
                y = segment[1] * cell_height
                color = (0, 255, 0) if i == 0 else (0, 180, 0)
                rect = pygame.Rect(x + 2, y + 2, cell_width - 4, cell_height - 4)
                pygame.draw.rect(surface, color, rect)
        
        # Draw food
        if hasattr(self.current_game, 'food') and self.current_game.food:
            x = self.current_game.food[0] * cell_width
            y = self.current_game.food[1] * cell_height
            rect = pygame.Rect(x + 2, y + 2, cell_width - 4, cell_height - 4)
            pygame.draw.rect(surface, (255, 0, 0), rect)
    
    def _draw_metrics(self):
        """Draw performance metrics using pygame"""
        if not self.current_metrics:
            return
        
        # Create metrics surface
        metrics_surface = pygame.Surface((self.stats_area.width - 4, self.stats_area.height - 4))
        metrics_surface.fill((40, 40, 40))
        
        # Draw score chart
        chart_rect = pygame.Rect(20, 20, 350, 150)
        self._draw_line_chart(metrics_surface, chart_rect, self.episode_scores, "Episode Scores")
        
        # Draw task usage pie chart
        if 'task_usage' in self.current_metrics:
            pie_rect = pygame.Rect(400, 20, 150, 150)
            self._draw_pie_chart(metrics_surface, pie_rect, self.current_metrics['task_usage'])
        
        # Draw current metrics text
        font = pygame.font.Font(None, 18)
        y = 180
        metrics_text = [
            f"Current Score: {self.current_metrics.get('score', 0):.1f}",
            f"Episode: {self.current_metrics.get('episode', 0)}",
            f"Steps: {self.current_metrics.get('steps', 0)}",
            f"Avg Score: {np.mean(self.episode_scores[-20:]) if self.episode_scores else 0:.1f}"
        ]
        
        for text in metrics_text:
            text_surface = font.render(text, True, (200, 200, 200))
            metrics_surface.blit(text_surface, (20, y))
            y += 20
        
        # Blit to main screen
        self.screen.blit(metrics_surface, (self.stats_area.x + 2, self.stats_area.y + 2))
    
    def _draw_line_chart(self, surface, rect, data, title):
        """Draw a simple line chart using pygame"""
        if not data:
            return
        
        # Draw border
        pygame.draw.rect(surface, (100, 100, 100), rect, 2)
        
        # Draw title
        font = pygame.font.Font(None, 16)
        title_text = font.render(title, True, (200, 200, 200))
        surface.blit(title_text, (rect.x + 5, rect.y - 20))
        
        # Draw chart
        if len(data) > 1:
            # Normalize data
            min_val = min(data)
            max_val = max(data)
            range_val = max_val - min_val if max_val != min_val else 1
            
            # Create points
            points = []
            for i, value in enumerate(data[-50:]):  # Last 50 points
                x = rect.x + int((i / (len(data[-50:]) - 1)) * rect.width)
                y = rect.y + rect.height - int(((value - min_val) / range_val) * rect.height)
                points.append((x, y))
            
            # Draw lines
            if len(points) > 1:
                pygame.draw.lines(surface, self.CHART_COLOR, False, points, 2)
            
            # Draw points
            for point in points[::5]:  # Every 5th point
                pygame.draw.circle(surface, self.CHART_COLOR, point, 3)
    
    def _draw_pie_chart(self, surface, rect, data):
        """Draw a simple pie chart using pygame"""
        if not data or sum(data.values()) == 0:
            return
        
        center = (rect.centerx, rect.centery)
        radius = min(rect.width, rect.height) // 2 - 10
        
        # Calculate angles
        total = sum(data.values())
        start_angle = 0
        
        for task, value in data.items():
            if value > 0:
                # Calculate sweep angle
                sweep_angle = (value / total) * 360
                
                # Get color
                color = self.TASK_COLORS.get(task, (150, 150, 150))
                
                # Draw arc (simplified - just draw lines)
                for angle in range(int(start_angle), int(start_angle + sweep_angle), 5):
                    rad = math.radians(angle)
                    x = center[0] + int(radius * math.cos(rad))
                    y = center[1] + int(radius * math.sin(rad))
                    pygame.draw.line(surface, color, center, (x, y), 2)
                
                start_angle += sweep_angle
        
        # Draw border
        pygame.draw.circle(surface, (100, 100, 100), center, radius, 2)
    
    def _update_stats_text(self):
        """Update statistics text box"""
        if not self.current_metrics:
            return
        
        stats_html = "<b>Current Performance:</b><br><br>"
        
        if 'score' in self.current_metrics:
            stats_html += f"Score: {self.current_metrics['score']:.1f}<br>"
        
        if 'episode' in self.current_metrics:
            stats_html += f"Episode: {self.current_metrics['episode']}<br>"
        
        if 'steps' in self.current_metrics:
            stats_html += f"Steps: {self.current_metrics['steps']}<br>"
        
        if self.episode_scores:
            avg_score = np.mean(self.episode_scores[-20:])
            stats_html += f"Avg Score (20): {avg_score:.1f}<br>"
            stats_html += f"Best Score: {max(self.episode_scores):.1f}<br>"
        
        if 'task_usage' in self.current_metrics:
            stats_html += "<br><b>Task Usage:</b><br>"
            for task, usage in self.current_metrics['task_usage'].items():
                stats_html += f"{task}: {usage:.1f}%<br>"
        
        stats_html += f"<br>Speed: {self.speed_slider.get_current_value():.1f}x"
        
        self.stats_text.html_text = stats_html
        self.stats_text.rebuild()
    
    def _start_game(self):
        """Start a new game session"""
        if self.game_thread and self.game_thread.is_alive():
            self.running = False
            self.game_thread.join()
        
        # Get selections
        game_name = self.game_dropdown.selected_option
        agent_type = self.agent_dropdown.selected_option
        
        # Create game
        if game_name == "Asteroids":
            self.current_game = AsteroidsGame(render=False)
        else:
            self.current_game = SnakeGame(render=False)
        
        # Reset game
        self.current_game.reset()
        
        # Create agent
        state_size = len(self.current_game.get_state_vector())
        action_size = self.current_game.get_action_space()
        
        if agent_type == "Meta-Learning":
            self.current_agent = MetaLearningAgent(state_size, action_size)
            self._load_models_for_agent()
        elif agent_type in ["Avoidance", "Combat", "Navigation"]:
            self.current_agent = create_task_agent(agent_type.lower(), state_size, action_size)
            model_path = f"models/best_{agent_type.lower()}_agent.pth"
            if os.path.exists(model_path):
                try:
                    self.current_agent.load(model_path)
                except:
                    pass
        elif agent_type == "Random":
            self.current_agent = "random"
        else:
            self.current_agent = "human"
        
        # Reset metrics
        self.episode_scores = []
        self.current_metrics = {}
        
        # Start game thread
        self.running = True
        self.paused = False
        self.game_thread = threading.Thread(target=self._game_loop)
        self.game_thread.start()
    
    def _game_loop(self):
        """Game loop running in separate thread"""
        episode = 0
        
        while self.running:
            if self.paused:
                continue
            
            state = self.current_game.reset()
            done = False
            total_reward = 0
            steps = 0
            task_usage = {}
            
            while not done and self.running and not self.paused:
                state_vector = self.current_game.get_state_vector()
                
                # Get action
                if self.current_agent == "random":
                    action = np.random.randint(0, self.current_game.get_action_space())
                elif self.current_agent == "human":
                    action = 0
                else:
                    action = self.current_agent.act(state_vector, training=False)
                    
                    # Track task usage
                    if hasattr(self.current_agent, 'current_task'):
                        task = self.current_agent.current_task
                        task_usage[task] = task_usage.get(task, 0) + 1
                
                # Step game
                next_state, reward, done, info = self.current_game.step(action)
                total_reward += reward
                steps += 1
                
                # Update metrics
                self.current_metrics = {
                    'score': total_reward,
                    'episode': episode,
                    'steps': steps,
                    'info': info
                }
                
                if task_usage:
                    total_usage = sum(task_usage.values())
                    task_percentages = {k: v/total_usage*100 for k, v in task_usage.items()}
                    self.current_metrics['task_usage'] = task_percentages
                
                # Control game speed
                speed = self.speed_slider.get_current_value()
                pygame.time.wait(int(16 / speed))
            
            # Episode complete
            self.episode_scores.append(total_reward)
            if len(self.episode_scores) > self.max_history:
                self.episode_scores.pop(0)
            
            episode += 1
    
    def _load_models_for_agent(self):
        """Load available models for current agent"""
        if not isinstance(self.current_agent, MetaLearningAgent):
            return
        
        loaded = 0
        for task in ['avoidance', 'combat', 'navigation']:
            for prefix in ['best_', '']:
                path = f"models/{prefix}{task}_agent.pth"
                if os.path.exists(path):
                    try:
                        self.current_agent.task_agents[task].load(path)
                        loaded += 1
                        break
                    except:
                        pass
        
        print(f"Loaded {loaded} task models")
    
    def _handle_ui_event(self, event):
        """Handle UI events"""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self._start_game()
            
            elif event.ui_element == self.pause_button:
                self.paused = not self.paused
                self.pause_button.set_text("Resume" if self.paused else "Pause")
    
    def run(self):
        """Run the dashboard"""
        running = True
        
        while running:
            time_delta = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                self._handle_ui_event(event)
                self.ui_manager.process_events(event)
            
            self.ui_manager.update(time_delta)
            
            # Draw everything
            self._draw_background()
            self._draw_game()
            self._draw_metrics()
            self._update_stats_text()
            
            self.ui_manager.draw_ui(self.screen)
            
            pygame.display.flip()
        
        # Cleanup
        self.running = False
        if self.game_thread:
            self.game_thread.join()
        if self.current_game:
            self.current_game.close()
        pygame.quit()


def main():
    dashboard = DashboardFixed()
    dashboard.run()


if __name__ == "__main__":
    main()