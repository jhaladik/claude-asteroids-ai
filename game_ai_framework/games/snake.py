"""
Snake Game Implementation V2
Following the correct BaseGame interface
"""

import pygame
import numpy as np
import random
from typing import Tuple, Dict, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_game import BaseGame, GameState, TaskContext


class SnakeGame(BaseGame):
    """Snake game implementation with AI-friendly interface"""
    
    def __init__(self, grid_size: int = 20, cell_size: int = 20, render: bool = True):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_size = grid_size * cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.DARK_GREEN = (0, 180, 0)
        
        # Initialize pygame if rendering
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()
        
        super().__init__(self.screen_size, self.screen_size, render)
    
    def reset(self) -> GameState:
        """Reset game to initial state"""
        # Initialize snake in center
        center = self.grid_size // 2
        self.snake = [(center, center), (center-1, center), (center-2, center)]
        self.direction = (1, 0)  # Right
        
        # Place initial food
        self._place_food()
        
        # Reset counters
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.lives = 1  # Snake has one life
        
        return self.get_state()
    
    def _place_food(self):
        """Place food randomly on empty cell"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))
        
        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            self.game_over = True
            self.food = None
    
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Execute one game step"""
        if self.game_over:
            return self.get_state(), 0, True, {"score": self.score}
        
        # Update direction based on action
        if action == 1:  # Turn left
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:  # Turn right
            self.direction = (self.direction[1], -self.direction[0])
        
        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.game_over = True
            self.lives = 0
            return self.get_state(), -10, True, {"score": self.score}
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            self.lives = 0
            return self.get_state(), -10, True, {"score": self.score}
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check food collision
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()  # Remove tail
            self.steps_since_food += 1
            reward = -0.01
            
            # Reward for moving closer to food
            if self.food:
                old_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
                new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
                if new_dist < old_dist:
                    reward += 0.1
        
        self.steps += 1
        
        # End game if taking too long
        if self.steps_since_food > self.grid_size * self.grid_size:
            self.game_over = True
            self.lives = 0
            reward = -5
        
        info = {
            "score": self.score,
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
            "snake_length": len(self.snake)
        }
        
        return self.get_state(), reward, self.game_over, info
    
    def render(self) -> None:
        """Render the game"""
        if not self.render_enabled:
            return
        
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = self.GREEN if i == 0 else self.DARK_GREEN
            rect = pygame.Rect(
                segment[0] * self.cell_size,
                segment[1] * self.cell_size,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw food
        if self.food:
            rect = pygame.Rect(
                self.food[0] * self.cell_size,
                self.food[1] * self.cell_size,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, self.RED, rect)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()
        if hasattr(self, 'clock'):
            self.clock.tick(10)
    
    def get_state(self) -> GameState:
        """Get current game state in universal format"""
        # Build entities list
        entities = []
        
        # Add snake segments
        for i, segment in enumerate(self.snake):
            entities.append({
                'type': 'snake_head' if i == 0 else 'snake_body',
                'position': (segment[0] / self.grid_size, segment[1] / self.grid_size),
                'index': i
            })
        
        # Add food
        if self.food:
            entities.append({
                'type': 'food',
                'position': (self.food[0] / self.grid_size, self.food[1] / self.grid_size)
            })
        
        # Calculate threats (walls and snake body)
        threats = []
        if self.snake:
            head = self.snake[0]
            
            # Check walls
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                next_x, next_y = head[0] + dx, head[1] + dy
                if next_x < 0 or next_x >= self.grid_size or next_y < 0 or next_y >= self.grid_size:
                    threats.append({
                        'type': 'wall',
                        'position': (next_x / self.grid_size, next_y / self.grid_size),
                        'distance': 1.0
                    })
            
            # Check snake body
            for segment in self.snake[1:]:
                dist = abs(head[0] - segment[0]) + abs(head[1] - segment[1])
                threats.append({
                    'type': 'snake_body',
                    'position': (segment[0] / self.grid_size, segment[1] / self.grid_size),
                    'distance': dist / self.grid_size
                })
        
        # Collectibles (food)
        collectibles = []
        if self.food and self.snake:
            head = self.snake[0]
            dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            collectibles.append({
                'type': 'food',
                'position': (self.food[0] / self.grid_size, self.food[1] / self.grid_size),
                'distance': dist / self.grid_size
            })
        
        # Player position
        player_pos = (self.snake[0][0] / self.grid_size, 
                     self.snake[0][1] / self.grid_size) if self.snake else (0.5, 0.5)
        
        state = GameState(
            frame=self._get_frame_array() if self.render_enabled else np.zeros((self.screen_size, self.screen_size, 3)),
            entities=entities,
            player_position=player_pos,
            threats=threats,
            collectibles=collectibles,
            score=self.score,
            lives=self.lives,
            level=1,
            time_remaining=None,
            distances_to_threats=[t['distance'] for t in threats],
            distances_to_collectibles=[c['distance'] for c in collectibles],
            safe_zones=self._calculate_safe_zones()
        )
        
        return state
    
    def _get_frame_array(self) -> np.ndarray:
        """Get current frame as numpy array"""
        if self.render_enabled and hasattr(self, 'screen'):
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
        return np.zeros((self.screen_size, self.screen_size, 3))
    
    def _calculate_safe_zones(self) -> List[Tuple[float, float]]:
        """Calculate safe positions on the grid"""
        safe_zones = []
        if not self.snake:
            return safe_zones
        
        head = self.snake[0]
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                x, y = head[0] + dx, head[1] + dy
                if (0 <= x < self.grid_size and 0 <= y < self.grid_size and
                    (x, y) not in self.snake):
                    safe_zones.append((x / self.grid_size, y / self.grid_size))
        
        return safe_zones
    
    def get_state_vector(self) -> np.ndarray:
        """Get state as vector for neural network"""
        if not self.snake:
            return np.zeros(18)
        
        head = self.snake[0]
        state = []
        
        # Food direction (normalized)
        if self.food:
            food_dx = (self.food[0] - head[0]) / self.grid_size
            food_dy = (self.food[1] - head[1]) / self.grid_size
        else:
            food_dx = food_dy = 0
        state.extend([food_dx, food_dy])
        
        # Danger detection (straight, left, right)
        for direction in [self.direction, 
                         (-self.direction[1], self.direction[0]),
                         (self.direction[1], -self.direction[0])]:
            next_pos = (head[0] + direction[0], head[1] + direction[1])
            danger = 0
            if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                next_pos[1] < 0 or next_pos[1] >= self.grid_size or
                next_pos in self.snake):
                danger = 1
            state.append(danger)
        
        # Current direction one-hot
        direction_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
        direction_onehot = [0, 0, 0, 0]
        if self.direction in direction_map:
            direction_onehot[direction_map[self.direction]] = 1
        state.extend(direction_onehot)
        
        # Wall distances
        wall_distances = [
            head[1] / self.grid_size,
            (self.grid_size - 1 - head[0]) / self.grid_size,
            (self.grid_size - 1 - head[1]) / self.grid_size,
            head[0] / self.grid_size
        ]
        state.extend(wall_distances)
        
        # Body distances
        body_distances = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            dist = 1.0
            for i in range(1, self.grid_size):
                check_pos = (head[0] + i*dx, head[1] + i*dy)
                if check_pos in self.snake[1:]:
                    dist = (i - 1) / self.grid_size
                    break
            body_distances.append(dist)
        state.extend(body_distances)
        
        # Snake length
        state.append(len(self.snake) / (self.grid_size * self.grid_size))
        
        return np.array(state, dtype=np.float32)
    
    def get_action_space(self) -> int:
        """Get number of possible actions"""
        return 3
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable action descriptions"""
        return ["Straight", "Turn Left", "Turn Right"]
    
    def analyze_task_context(self) -> TaskContext:
        """Analyze current game state for task priorities"""
        if not self.snake or not self.food:
            return TaskContext(
                avoidance_priority=0.0,
                collection_priority=0.0,
                combat_priority=0.0,
                navigation_priority=0.0,
                survival_priority=0.0
            )
        
        head = self.snake[0]
        
        # Check immediate danger
        danger_count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_pos = (head[0] + dx, head[1] + dy)
            if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                next_pos[1] < 0 or next_pos[1] >= self.grid_size or
                next_pos in self.snake):
                danger_count += 1
        
        avoidance_priority = min(danger_count / 2, 1.0)
        
        food_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        navigation_priority = min(food_dist / (self.grid_size * 2), 1.0)
        
        combat_priority = 0.0  # No combat in Snake
        
        survival_priority = len(self.snake) / (self.grid_size * self.grid_size)
        
        collection_priority = 0.8 if self.steps_since_food > 50 else 0.3
        
        return TaskContext(
            avoidance_priority=avoidance_priority,
            collection_priority=collection_priority,
            combat_priority=combat_priority,
            navigation_priority=navigation_priority,
            survival_priority=survival_priority
        )
    
    def get_semantic_features(self) -> Dict[str, float]:
        """Get high-level semantic features"""
        if not self.snake:
            return {
                "threat_level": 0,
                "resource_availability": 0,
                "space_constraint": 0,
                "progress_rate": 0
            }
        
        head = self.snake[0]
        
        # Threat level
        threats = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_pos = (head[0] + dx, head[1] + dy)
                if (check_pos[0] < 0 or check_pos[0] >= self.grid_size or
                    check_pos[1] < 0 or check_pos[1] >= self.grid_size or
                    check_pos in self.snake):
                    threats += 1
        
        threat_level = threats / 8.0
        resource_availability = 1.0 if self.food else 0.0
        space_constraint = len(self.snake) / (self.grid_size * self.grid_size)
        progress_rate = self.score / max(self.steps, 1) if self.steps > 0 else 0
        
        return {
            "threat_level": threat_level,
            "resource_availability": resource_availability,
            "space_constraint": space_constraint,
            "progress_rate": progress_rate
        }
    
    def close(self) -> None:
        """Clean up resources"""
        if self.render_enabled and hasattr(self, 'screen'):
            pygame.quit()


# Test the implementation
if __name__ == "__main__":
    game = SnakeGame(render=True)
    state = game.reset()
    
    print("Snake Game Test")
    print(f"State vector shape: {game.get_state_vector().shape}")
    print(f"Action space: {game.get_action_space()}")
    print(f"Actions: {game.get_action_meanings()}")
    
    # Run a few random steps
    for _ in range(100):
        action = random.randint(0, 2)
        state, reward, done, info = game.step(action)
        game.render()
        
        if done:
            print(f"Game Over! Score: {info['score']}")
            state = game.reset()
    
    game.close()