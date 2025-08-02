# AI Game Framework - Interface Guide

## Overview

The AI Game Framework provides multiple interfaces for interacting with the trained AI agents:

1. **Desktop Dashboard** - Full-featured pygame interface with real-time visualization
2. **Web Dashboard** - Browser-based interface for remote access
3. **Command Line Tools** - Direct access to training and demos

## Desktop Dashboard

The desktop dashboard provides a comprehensive interface with:

### Features
- Real-time game visualization
- Live performance metrics and charts
- Agent selection and control
- Model management (save/load)
- Speed control and pause functionality
- Task usage visualization

### Usage
```bash
cd game_ai_framework
python dashboard.py
```

### Controls
- **Game Selection**: Choose between Asteroids and Snake
- **Agent Selection**: Pick from Meta-Learning, task-specific agents, or random
- **Play/Pause**: Control game execution
- **Speed Slider**: Adjust game speed (0.1x to 5x)
- **Render Toggle**: Enable/disable visualization for performance

## Web Dashboard

The web dashboard provides a browser-based interface accessible from any device:

### Features
- Clean, modern web interface
- Real-time metrics updates
- Performance charts using Chart.js
- Task usage visualization
- Model status checking
- RESTful API for integration

### Usage
```bash
cd game_ai_framework
python web_dashboard.py
```
Then open http://localhost:5000 in your browser

### API Endpoints
- `GET /api/metrics` - Get current performance metrics
- `POST /api/start` - Start a new game session
- `POST /api/pause` - Pause/resume current game
- `POST /api/stop` - Stop current game
- `GET /api/models` - Check available models

## Launcher

The launcher provides easy access to all components:

```bash
python launcher.py
```

Options include:
1. Desktop Dashboard
2. Web Dashboard
3. Training Interface
4. Game Demos
5. Transfer Learning Demo
6. Model Status Check
7. Quick Training

## Interface Components

### Metrics Display
- **Score**: Current episode score
- **Episode**: Number of completed episodes
- **Steps**: Steps in current episode
- **Average Score**: Rolling average of last 20 episodes
- **Task Usage**: Percentage breakdown for meta-learning agent

### Visualization
- **Game View**: Live game rendering (desktop only)
- **Performance Chart**: Episode scores over time
- **Task Usage Bars**: Visual breakdown of task activation

### Model Management
- **Save Model**: Save current agent state
- **Load Model**: Load pre-trained models
- **Model Status**: Check which models are available

## Tips for Best Experience

1. **Performance**: Disable rendering for faster training
2. **Web Access**: Use web dashboard for remote monitoring
3. **Multiple Instances**: Run training and visualization separately
4. **Model Persistence**: Save models regularly during long training

## Customization

### Adding New Games
1. Implement the BaseGame interface
2. Add to game selection dropdown
3. Update API endpoints

### Adding New Visualizations
1. Extend metrics collection
2. Add new chart types
3. Update dashboard layout

### Custom Agents
1. Implement BaseAgent interface
2. Add to agent selection
3. Update model loading logic

## Troubleshooting

### Common Issues
- **PyGame not responding**: Check if another instance is running
- **Web dashboard not loading**: Ensure Flask is installed
- **Models not loading**: Check file paths and compatibility
- **Performance issues**: Disable rendering or use web interface

### Performance Optimization
- Use web dashboard for remote monitoring
- Disable game rendering during training
- Adjust game speed for testing
- Use batch training for multiple agents