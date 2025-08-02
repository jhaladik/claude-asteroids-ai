"""
Web-based Dashboard for AI Game Framework
Uses Flask for a browser-based interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request, Response
import json
import threading
import time
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from games.asteroids import AsteroidsGame
from games.snake import SnakeGame
from agents.meta_agent import MetaLearningAgent
from agents.task_agents import create_task_agent

app = Flask(__name__)

# Global state
game_state = {
    'current_game': None,
    'current_agent': None,
    'game_thread': None,
    'running': False,
    'paused': False,
    'metrics': {
        'score': 0,
        'episode': 0,
        'steps': 0,
        'episode_scores': [],
        'task_usage': {}
    }
}

@app.route('/')
def index():
    """Main dashboard page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Game Framework Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            .container {
                display: flex;
                height: 100vh;
            }
            .sidebar {
                width: 300px;
                background-color: #2d2d2d;
                padding: 20px;
                overflow-y: auto;
            }
            .main-content {
                flex: 1;
                padding: 20px;
                display: flex;
                flex-direction: column;
            }
            .game-area {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                text-align: center;
            }
            .metrics-area {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 20px;
                flex: 1;
            }
            .control-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #b0b0b0;
            }
            select, button {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #4d4d4d;
            }
            button.active {
                background-color: #5a5;
            }
            .metric {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                padding: 10px;
                background-color: #3d3d3d;
                border-radius: 4px;
            }
            .metric-label {
                color: #b0b0b0;
            }
            .metric-value {
                font-weight: bold;
                color: #5a5;
            }
            #performance-chart {
                width: 100%;
                height: 300px;
                margin-top: 20px;
            }
            h1, h2, h3 {
                color: #e0e0e0;
                margin-top: 0;
            }
            .status {
                padding: 5px 10px;
                border-radius: 4px;
                display: inline-block;
                margin-left: 10px;
            }
            .status.running {
                background-color: #5a5;
            }
            .status.stopped {
                background-color: #a55;
            }
            .task-usage {
                display: flex;
                margin-top: 10px;
            }
            .task-bar {
                height: 20px;
                transition: width 0.3s;
            }
            .task-bar.avoidance {
                background-color: #FF6B6B;
            }
            .task-bar.combat {
                background-color: #4ECDC4;
            }
            .task-bar.navigation {
                background-color: #45B7D1;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <h1>AI Game Framework</h1>
                
                <div class="control-group">
                    <label>Select Game:</label>
                    <select id="game-select">
                        <option value="asteroids">Asteroids</option>
                        <option value="snake">Snake</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Select Agent:</label>
                    <select id="agent-select">
                        <option value="meta">Meta-Learning</option>
                        <option value="avoidance">Avoidance</option>
                        <option value="combat">Combat</option>
                        <option value="navigation">Navigation</option>
                        <option value="random">Random</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <button id="play-button" onclick="startGame()">Start Game</button>
                    <button id="pause-button" onclick="pauseGame()">Pause</button>
                    <button id="stop-button" onclick="stopGame()">Stop</button>
                </div>
                
                <h3>Performance Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Score:</span>
                    <span class="metric-value" id="score">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Episode:</span>
                    <span class="metric-value" id="episode">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Steps:</span>
                    <span class="metric-value" id="steps">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Score:</span>
                    <span class="metric-value" id="avg-score">0</span>
                </div>
                
                <h3>Task Usage</h3>
                <div class="task-usage" id="task-usage">
                    <div class="task-bar avoidance" style="width: 33%"></div>
                    <div class="task-bar combat" style="width: 33%"></div>
                    <div class="task-bar navigation" style="width: 34%"></div>
                </div>
                
                <h3>Model Status</h3>
                <div id="model-status">
                    Loading...
                </div>
            </div>
            
            <div class="main-content">
                <div class="game-area">
                    <h2>Game View 
                        <span class="status stopped" id="game-status">Stopped</span>
                    </h2>
                    <p>Game visualization coming soon...</p>
                </div>
                
                <div class="metrics-area">
                    <h2>Performance Chart</h2>
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
        </div>
        
        <script>
            let chart = null;
            let updateInterval = null;
            
            // Initialize chart
            const ctx = document.getElementById('performance-chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Episode Score',
                        data: [],
                        borderColor: '#5a5',
                        backgroundColor: 'rgba(85, 170, 85, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: '#444'
                            },
                            ticks: {
                                color: '#b0b0b0'
                            }
                        },
                        x: {
                            grid: {
                                color: '#444'
                            },
                            ticks: {
                                color: '#b0b0b0'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#e0e0e0'
                            }
                        }
                    }
                }
            });
            
            function updateMetrics() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('score').textContent = data.score.toFixed(1);
                        document.getElementById('episode').textContent = data.episode;
                        document.getElementById('steps').textContent = data.steps;
                        document.getElementById('avg-score').textContent = data.avg_score.toFixed(1);
                        
                        // Update chart
                        if (data.episode_scores.length > 0) {
                            chart.data.labels = data.episode_scores.map((_, i) => i + 1);
                            chart.data.datasets[0].data = data.episode_scores;
                            chart.update();
                        }
                        
                        // Update task usage
                        if (data.task_usage) {
                            const total = Object.values(data.task_usage).reduce((a, b) => a + b, 0);
                            const taskBars = document.querySelectorAll('.task-bar');
                            let offset = 0;
                            ['avoidance', 'combat', 'navigation'].forEach((task, i) => {
                                const usage = data.task_usage[task] || 0;
                                const percentage = total > 0 ? (usage / total * 100) : 33.33;
                                taskBars[i].style.width = percentage + '%';
                            });
                        }
                        
                        // Update status
                        const statusElem = document.getElementById('game-status');
                        if (data.running) {
                            statusElem.textContent = data.paused ? 'Paused' : 'Running';
                            statusElem.className = 'status running';
                        } else {
                            statusElem.textContent = 'Stopped';
                            statusElem.className = 'status stopped';
                        }
                    });
            }
            
            function startGame() {
                const game = document.getElementById('game-select').value;
                const agent = document.getElementById('agent-select').value;
                
                fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({game: game, agent: agent})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        updateInterval = setInterval(updateMetrics, 1000);
                    }
                });
            }
            
            function pauseGame() {
                fetch('/api/pause', {method: 'POST'})
                    .then(response => response.json());
            }
            
            function stopGame() {
                fetch('/api/stop', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (updateInterval) {
                            clearInterval(updateInterval);
                            updateInterval = null;
                        }
                    });
            }
            
            // Check model status
            fetch('/api/models')
                .then(response => response.json())
                .then(data => {
                    const statusHtml = data.models.map(model => 
                        `<div class="metric">
                            <span class="metric-label">${model.name}:</span>
                            <span class="metric-value">${model.exists ? 'Loaded' : 'Not Found'}</span>
                        </div>`
                    ).join('');
                    document.getElementById('model-status').innerHTML = statusHtml;
                });
            
            // Initial update
            updateMetrics();
        </script>
    </body>
    </html>
    '''

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    metrics = game_state['metrics'].copy()
    metrics['running'] = game_state['running']
    metrics['paused'] = game_state['paused']
    
    # Calculate average score
    if metrics['episode_scores']:
        metrics['avg_score'] = np.mean(metrics['episode_scores'][-20:])
    else:
        metrics['avg_score'] = 0
    
    return jsonify(metrics)

@app.route('/api/start', methods=['POST'])
def start_game():
    """Start a new game"""
    data = request.json
    game_name = data.get('game', 'asteroids')
    agent_type = data.get('agent', 'meta')
    
    # Stop current game if running
    if game_state['running']:
        stop_game_internal()
    
    # Create game
    if game_name == 'asteroids':
        game_state['current_game'] = AsteroidsGame(render=False)
    else:
        game_state['current_game'] = SnakeGame(render=False)
    
    # Create agent
    state_size = len(game_state['current_game'].get_state_vector())
    action_size = game_state['current_game'].get_action_space()
    
    if agent_type == 'meta':
        game_state['current_agent'] = MetaLearningAgent(state_size, action_size)
        load_models_for_meta_agent()
    elif agent_type in ['avoidance', 'combat', 'navigation']:
        game_state['current_agent'] = create_task_agent(agent_type, state_size, action_size)
        try_load_model(agent_type)
    else:
        game_state['current_agent'] = 'random'
    
    # Reset metrics
    game_state['metrics'] = {
        'score': 0,
        'episode': 0,
        'steps': 0,
        'episode_scores': [],
        'task_usage': {}
    }
    
    # Start game thread
    game_state['running'] = True
    game_state['paused'] = False
    game_state['game_thread'] = threading.Thread(target=game_loop)
    game_state['game_thread'].start()
    
    return jsonify({'status': 'started'})

@app.route('/api/pause', methods=['POST'])
def pause_game():
    """Pause/resume game"""
    game_state['paused'] = not game_state['paused']
    return jsonify({'paused': game_state['paused']})

@app.route('/api/stop', methods=['POST'])
def stop_game():
    """Stop current game"""
    stop_game_internal()
    return jsonify({'status': 'stopped'})

@app.route('/api/models')
def check_models():
    """Check which models are available"""
    models = []
    for agent in ['avoidance', 'combat', 'navigation', 'meta']:
        path = f"models/best_{agent}_agent.pth"
        models.append({
            'name': agent.capitalize(),
            'exists': os.path.exists(path)
        })
    return jsonify({'models': models})

def game_loop():
    """Main game loop"""
    episode = 0
    
    while game_state['running']:
        if game_state['paused']:
            time.sleep(0.1)
            continue
        
        state = game_state['current_game'].reset()
        done = False
        total_reward = 0
        steps = 0
        task_usage = {'avoidance': 0, 'combat': 0, 'navigation': 0}
        
        while not done and game_state['running'] and not game_state['paused']:
            state_vector = game_state['current_game'].get_state_vector()
            
            # Get action
            if game_state['current_agent'] == 'random':
                action = np.random.randint(0, game_state['current_game'].get_action_space())
            else:
                action = game_state['current_agent'].act(state_vector, training=False)
                
                # Track task usage
                if hasattr(game_state['current_agent'], 'current_task'):
                    task = game_state['current_agent'].current_task
                    if task in task_usage:
                        task_usage[task] += 1
            
            # Step game
            next_state, reward, done, info = game_state['current_game'].step(action)
            total_reward += reward
            steps += 1
            
            # Update metrics
            game_state['metrics']['score'] = total_reward
            game_state['metrics']['episode'] = episode
            game_state['metrics']['steps'] = steps
            game_state['metrics']['task_usage'] = task_usage
            
            # Control speed
            time.sleep(0.016)  # ~60 FPS
        
        # Episode complete
        game_state['metrics']['episode_scores'].append(total_reward)
        if len(game_state['metrics']['episode_scores']) > 50:
            game_state['metrics']['episode_scores'].pop(0)
        
        episode += 1

def stop_game_internal():
    """Internal function to stop game"""
    game_state['running'] = False
    if game_state['game_thread']:
        game_state['game_thread'].join()
    if game_state['current_game']:
        game_state['current_game'].close()

def load_models_for_meta_agent():
    """Load models for meta-learning agent"""
    if not isinstance(game_state['current_agent'], MetaLearningAgent):
        return
    
    for task in ['avoidance', 'combat', 'navigation']:
        for prefix in ['best_', '']:
            path = f"models/{prefix}{task}_agent.pth"
            if os.path.exists(path):
                try:
                    game_state['current_agent'].task_agents[task].load(path)
                    break
                except:
                    pass

def try_load_model(agent_type):
    """Try to load model for specific agent"""
    path = f"models/best_{agent_type}_agent.pth"
    if os.path.exists(path):
        try:
            game_state['current_agent'].load(path)
        except:
            pass

if __name__ == '__main__':
    print("Starting Web Dashboard on http://localhost:5000")
    app.run(debug=True, use_reloader=False)