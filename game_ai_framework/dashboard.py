"""
AI Game Framework Dashboard
Interactive interface for managing and visualizing AI agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import pygame_gui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json
from datetime import datetime
import threading
from typing import Dict, List, Optional

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
from training.train_agents import train_task_agent, TrainingLogger


class Dashboard:
    """Main dashboard interface"""
    
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
        
        self.train_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, 260, 280, 40),
            text="Train Agent",
            manager=self.ui_manager
        )
        
        # Model Management
        self.model_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 320, 280, 30),
            text="Model Management:",
            manager=self.ui_manager
        )
        
        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, 350, 135, 40),
            text="Save Model",
            manager=self.ui_manager
        )
        
        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(155, 350, 135, 40),
            text="Load Model",
            manager=self.ui_manager
        )
        
        # Performance Stats
        self.stats_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 410, 280, 30),
            text="Performance Stats:",
            manager=self.ui_manager
        )
        
        self.stats_text = pygame_gui.elements.UITextBox(
            html_text="No game running",
            relative_rect=pygame.Rect(10, 440, 280, 200),
            manager=self.ui_manager
        )
        
        # Settings
        self.settings_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 660, 280, 30),
            text="Settings:",
            manager=self.ui_manager
        )
        
        self.render_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, 690, 280, 30),
            text="[X] Render Game",
            manager=self.ui_manager
        )
        self.render_enabled = True
        
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 730, 100, 30),
            text="Game Speed:",
            manager=self.ui_manager
        )
        
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(120, 730, 170, 30),
            start_value=1.0,
            value_range=(0.1, 5.0),
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
        if self.current_game and self.render_enabled:
            # Get game surface
            if hasattr(self.current_game, 'screen'):
                game_surface = self.current_game.screen
                # Scale to fit game area
                scaled_surface = pygame.transform.scale(
                    game_surface, 
                    (self.game_area.width - 4, self.game_area.height - 4)
                )
                self.screen.blit(scaled_surface, 
                               (self.game_area.x + 2, self.game_area.y + 2))
        else:
            # Draw placeholder
            font = pygame.font.Font(None, 36)
            text = font.render("No Game Running", True, self.TEXT_COLOR)
            text_rect = text.get_rect(center=self.game_area.center)
            self.screen.blit(text, text_rect)
    
    def _draw_metrics(self):
        """Draw performance metrics"""
        if not self.current_metrics:
            return
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.4), facecolor='#323232')
        
        # Score history
        if self.episode_scores:
            ax1.plot(self.episode_scores[-50:], color='#64C864')
            ax1.set_title('Episode Scores', color='white', fontsize=10)
            ax1.set_xlabel('Episode', color='white', fontsize=8)
            ax1.set_ylabel('Score', color='white', fontsize=8)
            ax1.tick_params(colors='white', labelsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#323232')
        
        # Task usage pie chart
        if 'task_usage' in self.current_metrics:
            task_usage = self.current_metrics['task_usage']
            if task_usage:
                labels = list(task_usage.keys())
                values = list(task_usage.values())
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                       textprops={'color': 'white', 'fontsize': 8})
                ax2.set_title('Task Usage', color='white', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.screen.blit(surf, (self.stats_area.x + 2, self.stats_area.y + 2))
        
        plt.close(fig)
    
    def _update_stats_text(self):
        """Update statistics text box"""
        if not self.current_metrics:
            return
        
        stats_html = "<b>Current Performance:</b><br>"
        
        if 'score' in self.current_metrics:
            stats_html += f"Score: {self.current_metrics['score']:.1f}<br>"
        
        if 'episode' in self.current_metrics:
            stats_html += f"Episode: {self.current_metrics['episode']}<br>"
        
        if 'steps' in self.current_metrics:
            stats_html += f"Steps: {self.current_metrics['steps']}<br>"
        
        if 'epsilon' in self.current_metrics:
            stats_html += f"Exploration: {self.current_metrics['epsilon']:.3f}<br>"
        
        if 'avg_score' in self.current_metrics:
            stats_html += f"Avg Score: {self.current_metrics['avg_score']:.1f}<br>"
        
        if 'task_usage' in self.current_metrics:
            stats_html += "<br><b>Task Usage:</b><br>"
            for task, usage in self.current_metrics['task_usage'].items():
                stats_html += f"{task}: {usage:.1f}%<br>"
        
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
            self.current_game = AsteroidsGame(render=self.render_enabled)
        else:
            self.current_game = SnakeGame(render=self.render_enabled)
        
        # Create agent
        state_size = len(self.current_game.get_state_vector())
        action_size = self.current_game.get_action_space()
        
        if agent_type == "Meta-Learning":
            self.current_agent = MetaLearningAgent(state_size, action_size)
            # Try to load models
            self._load_models_for_agent()
        elif agent_type in ["Avoidance", "Combat", "Navigation"]:
            from agents.task_agents import create_task_agent
            self.current_agent = create_task_agent(agent_type.lower(), 
                                                  state_size, action_size)
            # Try to load specific model
            model_path = f"models/best_{agent_type.lower()}_agent.pth"
            if os.path.exists(model_path):
                self.current_agent.load(model_path)
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
                    # TODO: Implement human control
                    action = 0
                else:
                    action = self.current_agent.act(state_vector, training=False)
                    
                    # Track task usage for meta-agent
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
                pygame.time.wait(int(16 / speed))  # ~60 FPS adjusted by speed
            
            # Episode complete
            self.episode_scores.append(total_reward)
            if len(self.episode_scores) > 100:
                self.episode_scores.pop(0)
            
            self.current_metrics['avg_score'] = np.mean(self.episode_scores[-20:])
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
            
            elif event.ui_element == self.train_button:
                # TODO: Implement training interface
                print("Training interface not yet implemented")
            
            elif event.ui_element == self.save_button:
                if self.current_agent and hasattr(self.current_agent, 'save'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"models/dashboard_save_{timestamp}.pth"
                    self.current_agent.save(path)
                    print(f"Model saved to {path}")
            
            elif event.ui_element == self.load_button:
                # TODO: Implement file dialog
                print("Load dialog not yet implemented")
            
            elif event.ui_element == self.render_checkbox:
                self.render_enabled = not self.render_enabled
                self.render_checkbox.set_text(
                    "[X] Render Game" if self.render_enabled else "[ ] Render Game"
                )
    
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
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()