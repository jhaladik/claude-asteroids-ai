# Claude Asteroids AI

An advanced AI system for playing Asteroids using Deep Q-Learning with meta-learning capabilities and modular task decomposition.

## Project Structure

```
claude-asteroids-ai/
├── asteroids_ai_v2.py          # Main game with integrated AI and logging
├── ai_parameters.json          # Configurable AI parameters
├── analyze_training_logs.py    # Training analysis tools
├── compare_training_params.py  # Parameter comparison utility
├── game_ai_framework/          # Modular framework for transfer learning
│   ├── core/                   # Base interfaces
│   ├── games/                  # Game implementations
│   ├── agents/                 # AI agents
│   └── main.py                # Framework demo
└── training_logs/              # Training session logs
```

## Quick Start

### Run Training with Different Parameters
```bash
python asteroids_ai_v2.py
# Select parameter set from menu (default, fast_learning, balanced_learning)
```

### Analyze Training Results
```bash
python analyze_training_logs.py
python compare_training_params.py
```

### Run Modular Framework Demo
```bash
cd game_ai_framework
python main.py
```

## Key Features

- **Deep Q-Network (DQN)** with experience replay
- **Meta-Learning** with 5 specialized networks
- **Task Decomposition** into avoidance, combat, and navigation
- **Configurable Parameters** for experimentation
- **Comprehensive Logging** for performance analysis
- **Transfer Learning** framework for multiple games

## AI Architecture

### DQN Agent
- Neural network with 3 hidden layers
- Experience replay buffer (10,000 samples)
- Epsilon-greedy exploration
- Target network for stability

### Meta-Learning Components
1. **Threat Assessment Network**: Evaluates immediate dangers
2. **Movement Planning Network**: Optimizes positioning
3. **Combat Strategy Network**: Handles targeting and shooting
4. **Resource Management Network**: Manages bullets and positioning
5. **Survival Priority Network**: Balances risk vs reward

### Task-Specific Agents
- **AvoidanceAgent**: Specialized in dodging asteroids
- **CombatAgent**: Focused on destroying targets
- **NavigationAgent**: Optimizes movement patterns

## Parameter Sets

### Default (Slow Learning)
- Learning rate: 0.001
- Epsilon decay: 0.9998
- Best for: Thorough exploration

### Fast Learning
- Learning rate: 0.001
- Epsilon decay: 0.99
- Best for: Quick convergence

### Balanced Learning (Recommended)
- Learning rate: 0.0005
- Epsilon decay: 0.9995
- Best for: Optimal performance

## Performance Metrics

Training tracks:
- Episode scores and survival time
- Q-value evolution
- Loss progression
- Action distribution
- Exploration rate

## Development History

1. Started with basic DQN implementation
2. Added meta-learning with task networks
3. Discovered and fixed epsilon decay issues
4. Created modular framework for transfer learning
5. Implemented comprehensive logging and analysis

## Future Enhancements

- [ ] Automated hyperparameter tuning
- [ ] Real-time performance dashboard
- [ ] Additional game implementations
- [ ] Online learning capabilities
- [ ] Multi-agent coordination

## Requirements

- Python 3.8+
- PyTorch
- Pygame
- NumPy
- Matplotlib (for analysis)

## License

This project was developed as an educational exploration of AI techniques in game playing.