"""
Asteroids Game Implementation
Compatible with the meta-learning framework
"""

import pygame
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import sys
sys.path.append('..')
from core.base_game import BaseGame, GameState, TaskContext


# Game constants
WIDTH = 800
HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


@dataclass
class Ship:
    """Player ship"""
    x: float
    y: float
    angle: float = 0
    vel_x: float = 0
    vel_y: float = 0
    radius: int = 10
    
    def update(self):
        """Update ship position"""
        self.x = (self.x + self.vel_x) % WIDTH
        self.y = (self.y + self.vel_y) % HEIGHT
        self.vel_x *= 0.99  # friction
        self.vel_y *= 0.99


@dataclass  
class Asteroid:
    """Asteroid entity"""
    x: float
    y: float
    vel_x: float
    vel_y: float
    size: int
    color: Tuple[int, int, int]
    
    def update(self):
        """Update asteroid position"""
        self.x = (self.x + self.vel_x) % WIDTH
        self.y = (self.y + self.vel_y) % HEIGHT


@dataclass
class Bullet:
    """Bullet entity"""
    x: float
    y: float
    vel_x: float
    vel_y: float
    lifetime: int = 60
    
    def update(self):
        """Update bullet position"""
        self.x += self.vel_x
        self.y += self.vel_y
        self.lifetime -= 1


class AsteroidsGame(BaseGame):
    """Asteroids game implementation"""
    
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, render: bool = True):
        self.width = width
        self.height = height
        self.render_enabled = render
        
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Asteroids - Meta-Learning Framework")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.reset()
    
    def reset(self) -> GameState:
        """Reset game to initial state"""
        self.ship = Ship(x=self.width/2, y=self.height/2)
        self.asteroids = []
        self.bullets = []
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.frame_count = 0
        
        # Create initial asteroids
        for _ in range(3):
            self._spawn_asteroid()
        
        return self.get_state()
    
    def _spawn_asteroid(self):
        """Spawn a new asteroid at screen edge"""
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            x, y = random.randint(0, self.width), 0
        elif side == 'bottom':
            x, y = random.randint(0, self.width), self.height
        elif side == 'left':
            x, y = 0, random.randint(0, self.height)
        else:
            x, y = self.width, random.randint(0, self.height)
        
        # Velocity toward center with some randomness
        center_x, center_y = self.width/2, self.height/2
        angle = math.atan2(center_y - y, center_x - x) + random.uniform(-0.5, 0.5)
        speed = random.uniform(1, 3)
        
        asteroid = Asteroid(
            x=x, y=y,
            vel_x=speed * math.cos(angle),
            vel_y=speed * math.sin(angle),
            size=random.randint(20, 50),
            color=random.choice([RED, BLUE, YELLOW])
        )
        self.asteroids.append(asteroid)
    
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Execute one game step"""
        self.frame_count += 1
        reward = 0.1  # Small reward for surviving
        
        # Handle actions: 0=nothing, 1=left, 2=right, 3=thrust, 4=fire
        if action == 1:  # Rotate left
            self.ship.angle -= 5
        elif action == 2:  # Rotate right  
            self.ship.angle += 5
        elif action == 3:  # Thrust
            thrust = 0.5
            self.ship.vel_x += thrust * math.cos(math.radians(self.ship.angle))
            self.ship.vel_y += thrust * math.sin(math.radians(self.ship.angle))
        elif action == 4:  # Fire
            if len(self.bullets) < 5:  # Limit bullets
                self._fire_bullet()
        
        # Update entities
        self.ship.update()
        
        for asteroid in self.asteroids[:]:
            asteroid.update()
            
            # Check ship collision
            dist = math.hypot(asteroid.x - self.ship.x, asteroid.y - self.ship.y)
            if dist < asteroid.size + self.ship.radius:
                self.lives -= 1
                reward = -10
                if self.lives <= 0:
                    self.game_over = True
                # Reset ship position
                self.ship = Ship(x=self.width/2, y=self.height/2)
        
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.lifetime <= 0 or self._out_of_bounds(bullet.x, bullet.y):
                self.bullets.remove(bullet)
                continue
            
            # Check bullet-asteroid collisions
            for asteroid in self.asteroids[:]:
                dist = math.hypot(asteroid.x - bullet.x, asteroid.y - bullet.y)
                if dist < asteroid.size:
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    self.score += 10
                    reward += 5
                    break
        
        # Spawn new asteroids
        if len(self.asteroids) < 3:
            self._spawn_asteroid()
        
        # Get next state
        next_state = self.get_state()
        
        info = {
            'lives': self.lives,
            'asteroids_destroyed': self.score // 10,
            'frame': self.frame_count
        }
        
        return next_state, reward, self.game_over, info
    
    def _fire_bullet(self):
        """Fire a bullet from ship"""
        speed = 10
        bullet = Bullet(
            x=self.ship.x,
            y=self.ship.y,
            vel_x=speed * math.cos(math.radians(self.ship.angle)),
            vel_y=speed * math.sin(math.radians(self.ship.angle))
        )
        self.bullets.append(bullet)
    
    def _out_of_bounds(self, x: float, y: float) -> bool:
        """Check if position is out of screen bounds"""
        return x < 0 or x > self.width or y < 0 or y > self.height
    
    def render(self) -> None:
        """Render the game"""
        if not self.render_enabled:
            return
        
        self.screen.fill(BLACK)
        
        # Draw ship
        ship_points = [
            (self.ship.x + 10 * math.cos(math.radians(self.ship.angle)),
             self.ship.y + 10 * math.sin(math.radians(self.ship.angle))),
            (self.ship.x + 10 * math.cos(math.radians(self.ship.angle + 140)),
             self.ship.y + 10 * math.sin(math.radians(self.ship.angle + 140))),
            (self.ship.x + 10 * math.cos(math.radians(self.ship.angle - 140)),
             self.ship.y + 10 * math.sin(math.radians(self.ship.angle - 140)))
        ]
        pygame.draw.polygon(self.screen, WHITE, ship_points)
        
        # Draw asteroids
        for asteroid in self.asteroids:
            pygame.draw.circle(self.screen, asteroid.color, 
                             (int(asteroid.x), int(asteroid.y)), asteroid.size)
        
        # Draw bullets
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, GREEN, 
                             (int(bullet.x), int(bullet.y)), 3)
        
        # Draw UI
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.lives}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 50))
        
        pygame.display.flip()
        if hasattr(self, 'clock'):
            self.clock.tick(FPS)
    
    def get_state(self) -> GameState:
        """Get current game state in universal format"""
        # Get threats (asteroids)
        threats = []
        for asteroid in self.asteroids:
            threats.append({
                'x': asteroid.x / self.width,  # Normalize to 0-1
                'y': asteroid.y / self.height,
                'vel_x': asteroid.vel_x / 5,  # Normalize velocities
                'vel_y': asteroid.vel_y / 5,
                'size': asteroid.size / 50,
                'distance': math.hypot(asteroid.x - self.ship.x, 
                                     asteroid.y - self.ship.y) / self.width
            })
        
        # Sort by distance
        threats.sort(key=lambda t: t['distance'])
        
        # Get entities list
        entities = [{
            'type': 'player',
            'x': self.ship.x / self.width,
            'y': self.ship.y / self.height,
            'vel_x': self.ship.vel_x / 10,
            'vel_y': self.ship.vel_y / 10,
            'angle': self.ship.angle / 360
        }]
        
        for asteroid in self.asteroids:
            entities.append({
                'type': 'threat',
                'x': asteroid.x / self.width,
                'y': asteroid.y / self.height,
                'vel_x': asteroid.vel_x / 5,
                'vel_y': asteroid.vel_y / 5,
                'size': asteroid.size / 50
            })
        
        state = GameState(
            frame=self._get_frame_array() if self.render_enabled else np.zeros((600, 800, 3)),
            entities=entities,
            player_position=(self.ship.x / self.width, self.ship.y / self.height),
            threats=threats,
            collectibles=[],  # No collectibles in asteroids
            score=self.score,
            lives=self.lives,
            level=1,
            time_remaining=None,
            distances_to_threats=[t['distance'] for t in threats],
            distances_to_collectibles=[],
            safe_zones=self._calculate_safe_zones()
        )
        
        return state
    
    def _get_frame_array(self) -> np.ndarray:
        """Get current frame as numpy array"""
        if self.render_enabled and hasattr(self, 'screen'):
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def _calculate_safe_zones(self) -> List[Tuple[float, float]]:
        """Calculate relatively safe positions on screen"""
        # Simple grid-based approach
        safe_zones = []
        grid_size = 100
        
        for x in range(0, self.width, grid_size):
            for y in range(0, self.height, grid_size):
                # Check if any asteroid is nearby
                is_safe = True
                for asteroid in self.asteroids:
                    dist = math.hypot(asteroid.x - x, asteroid.y - y)
                    if dist < asteroid.size + 50:  # Safety margin
                        is_safe = False
                        break
                
                if is_safe:
                    safe_zones.append((x / self.width, y / self.height))
        
        return safe_zones
    
    def get_action_space(self) -> int:
        """Return number of possible actions"""
        return 5  # nothing, left, right, thrust, fire
    
    def get_action_meanings(self) -> List[str]:
        """Return human-readable action descriptions"""
        return ['Nothing', 'Rotate Left', 'Rotate Right', 'Thrust', 'Fire']
    
    def analyze_task_context(self) -> TaskContext:
        """Analyze current state to determine task priorities"""
        state = self.get_state()
        
        # Calculate task priorities based on game state
        nearest_threat_dist = state.distances_to_threats[0] if state.distances_to_threats else 1.0
        
        # High avoidance priority when threats are close
        avoidance_priority = max(0, 1 - nearest_threat_dist * 2)
        
        # Combat priority when threats are at medium distance
        combat_priority = 1 - abs(nearest_threat_dist - 0.3) * 3
        combat_priority = max(0, combat_priority)
        
        # Always some survival priority
        survival_priority = 0.5 + 0.5 * (1 - self.lives / 3)
        
        # Navigation to safe zones when needed
        navigation_priority = 0.3 if avoidance_priority > 0.7 else 0.1
        
        return TaskContext(
            avoidance_priority=avoidance_priority,
            collection_priority=0.0,  # No collection in asteroids
            combat_priority=combat_priority,
            navigation_priority=navigation_priority,
            survival_priority=survival_priority
        )
    
    def get_state_vector(self) -> np.ndarray:
        """Get state as fixed-size vector for neural networks"""
        state = self.get_state()
        
        # Basic features
        features = [
            state.player_position[0],
            state.player_position[1],
            self.ship.vel_x / 10,
            self.ship.vel_y / 10,
            math.cos(math.radians(self.ship.angle)),
            math.sin(math.radians(self.ship.angle)),
            len(self.bullets) / 5,
            self.lives / 3
        ]
        
        # Add nearest N asteroids
        max_asteroids = 5
        for i in range(max_asteroids):
            if i < len(state.threats):
                threat = state.threats[i]
                features.extend([
                    threat['x'],
                    threat['y'],
                    threat['vel_x'],
                    threat['vel_y'],
                    threat['size'],
                    threat['distance']
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 1])  # Padding
        
        return np.array(features, dtype=np.float32)
    
    def get_semantic_features(self) -> Dict[str, float]:
        """Get high-level semantic features"""
        state = self.get_state()
        context = self.analyze_task_context()
        
        return {
            'threat_level': context.avoidance_priority,
            'combat_opportunity': context.combat_priority,
            'escape_routes': len(state.safe_zones) / 10,
            'resource_availability': len(self.bullets) / 5,
            'health_status': self.lives / 3,
            'spatial_density': len(self.asteroids) / 10,
            'momentum': math.hypot(self.ship.vel_x, self.ship.vel_y) / 10
        }
    
    def close(self) -> None:
        """Clean up resources"""
        if self.render_enabled:
            pygame.quit()