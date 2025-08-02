# Game AI Framework - Modular Meta-Learning System

A modular framework for training AI agents that can learn and transfer skills across multiple games.

## Architecture Overview

```
game_ai_framework/
├── core/                    # Core interfaces and base classes
│   ├── base_game.py        # Universal game interface
│   └── base_agent.py       # Agent interfaces
├── games/                   # Game implementations
│   └── asteroids.py        # Asteroids game (example)
├── agents/                  # AI agent implementations
│   ├── task_agents.py      # Task-specific agents
│   └── meta_agent.py       # Meta-learning coordinator
├── training/               # Training utilities (TODO)
└── utils/                  # Helper functions (TODO)
```

## Key Concepts

### 1. Universal Game Interface
All games implement the `BaseGame` interface, providing:
- Standardized state representation (`GameState`)
- Task context analysis (`TaskContext`)
- Semantic features that transfer across games

### 2. Task-Specific Agents
Specialized agents for different game skills:
- **AvoidanceAgent**: Dodging threats and staying safe
- **CombatAgent**: Targeting and shooting enemies
- **NavigationAgent**: Movement and positioning

### 3. Meta-Learning Agent
Coordinates multiple task agents:
- Dynamically selects appropriate task agent based on game state
- Tracks performance and adjusts task importance
- Facilitates knowledge transfer between games

### 4. State Representation
Three levels of state abstraction:
1. **Raw state**: Game-specific vector
2. **Semantic features**: High-level transferable features
3. **Task context**: Current task priorities

## Usage

### Basic Demo
```bash
cd game_ai_framework
python main.py
```

### Train Individual Task Agents
```bash
python main.py train_tasks
```

## Extending the Framework

### Adding a New Game
1. Create a new file in `games/`
2. Implement the `BaseGame` interface
3. Define state vector representation
4. Implement task context analysis

### Adding a New Task Agent
1. Create a new class inheriting from `TaskSpecificAgent`
2. Implement confidence scoring
3. Add task-specific reward shaping
4. Register in the factory function

## Example: Asteroids

The framework includes Asteroids as an example implementation showing:
- How to wrap existing game logic
- State vector design for neural networks
- Task context analysis
- Semantic feature extraction

## Transfer Learning Workflow

1. **Train on Game A**: Task agents learn specialized skills
2. **Analyze Game B**: Identify which tasks are relevant
3. **Transfer Knowledge**: Use trained task agents on new game
4. **Fine-tune**: Adapt to game-specific differences

## Key Features

- **Modular Design**: Easy to add new games and agents
- **Task Decomposition**: Complex behaviors from simple tasks
- **Transfer Learning**: Reuse learned skills across games
- **Dynamic Coordination**: Meta-agent adapts to game context
- **Extensible**: Built for experimentation

## Future Enhancements

- [ ] Training pipeline with logging
- [ ] Task detection from gameplay observation
- [ ] Multi-game training curriculum
- [ ] Performance benchmarking suite
- [ ] Visualization tools
- [ ] More game implementations (Snake, Pong, etc.)

## Design Principles

1. **Separation of Concerns**: Games, agents, and training are independent
2. **Reusability**: Task agents work across multiple games
3. **Flexibility**: Easy to experiment with different architectures
4. **Scalability**: Add games and tasks without changing core code