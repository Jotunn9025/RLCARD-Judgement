# RLCard-Judgement

A custom RLCARD environment for the Judgement (Oh Hell) card game, featuring Neural Fictitious Self-Play (NFSP) integration and advanced RL training pipelines.

## Installation

### Using `uv` (Recommended)

If you have `uv` installed, it's the fastest way to get started:

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` in a virtual environment.

To run commands in the environment:
```bash
uv run python train_nfsp.py
uv run pytest
```

### Using `pip`

If you prefer traditional pip:

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install rlcard torch pytest
```

### Python Requirements
- Python 3.10 or higher
- PyTorch 2.10.0+
- RLCard 1.2.0+

## The Judgement Environment

**Judgement** (also known as "Oh Hell") is a 4-player trick-taking card game implemented as a custom RLCard environment. It's designed for training deep RL agents in cooperative and competitive multi-agent settings.

### Game Flow

1. **Dealing Phase**: Cards are dealt to all 4 players (from 13 cards down to 1, then back to 13).
2. **Bidding Phase**: Each player bids how many tricks they expect to win in the current round. The constraint is that the total bids cannot equal the number of cards in play (to prevent perfect prediction).
3. **Playing Phase**: 
   - Players play one card per trick in turn order
   - Must follow the lead suit if possible
   - Highest trump wins; otherwise highest card of the lead suit wins
4. **Scoring**: 
   - **Success**: Win exactly your bid → `10 + bid` points
   - **Failure**: Win more or fewer tricks than bid → `-10 × |bid - tricks_won|` points

### Observation Space (227 Features)

The environment provides a compact state representation for agents:

| Component | Size | Description |
|-----------|------|-------------|
| My Hand | 52 bits | One-hot encoding of cards in hand |
| Trump Suit | 4 bits | One-hot encoding of current trump suit |
| Current Trick | 52 bits | One-hot encoding of cards played this trick |
| Bids | 4 floats | Normalized bids of all 4 players (divided by current round's card count) |
| Tricks Won | 4 floats | Normalized tricks won so far (divided by current round's card count) |
| Dealer Position | 4 bits | One-hot encoding of dealer |
| Phase Indicator | 1 bit | 0 = Bidding, 1 = Playing |
| My Bid | 1 float | Normalized personal bid (divided by current round's card count) |
| My Tricks Won | 1 float | Normalized personal tricks won (divided by current round's card count) |
| Trick Winners History | 52 bits | One-hot per trick (13 tricks × 4 players) |
| Played Cards History | 52 bits | Bitmask of all cards played so far |
| **Total** | **227 features** | |

### Action Space

- **Actions 0-13**: Bid for 0 to 13 tricks (during bidding phase)
- **Actions 14-65**: Play specific card (card index + 14, during playing phase)

## NFSP Agent Training

**Neural Fictitious Self-Play (NFSP)** is an end-to-end RL algorithm designed to compute approximate Nash equilibria in imperfect-information games through self-play. The algorithm maintains two networks:

1. **Q-Network (RL Branch)**: Learns the best response to the current average strategy via Q-learning
2. **Policy Network (SL Branch)**: Tracks the empirical distribution of actions (average strategy)

### Training Process

The NFSP training loop consists of:

1. **Self-Play Episodes**: All 4 agents play against each other for one game
2. **Data Collection**: Trajectories are collected during each episode
3. **Network Training**: 
   - Q-network updates via RL loss (DQN with experience replay)
   - Policy network updates via supervised learning on action history
4. **Periodic Evaluation**: Agent 0 is evaluated against 3 random agents to measure progress

### Running Training

```bash
# Default settings (13 cards, 50k episodes)
uv run python train_nfsp.py

# Custom configuration
uv run python train_nfsp.py --episodes 100000 --cards 10 --sl_lr 0.01 --evaluate_every 500
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 50000 | Total training episodes |
| `--cards` | 13 | Starting number of cards per player |
| `--sl_lr` | 0.005 | Supervised learning (policy) learning rate |
| `--evaluate_every` | 500 | Evaluation interval (episodes) |
| `--evaluate_num` | 100 | Number of games per evaluation |
| `--seed` | 42 | Random seed for reproducibility |
| `--save_dir` | nfsp_checkpoints | Directory to save trained weights |

### Output & Checkpoints

Trained agent weights are saved as `.pth` files (one per player). The training loop prints:
- Episode number
- Average payoff vs random agents
- Average payoff in self-play
- RL loss and SL loss (network training metrics)

## Game Engine & Reward System Improvements

### 1. **Reward Signal Improvements (Soft Penalty Architecture)**

*The original payoff function was binary (win or lose the bid), which created sparse rewards that penalized 'near-misses' identically to complete failures. This has been resolved.*

**Current Reward Architecture**:
The environment has been refactored to use a two-tiered reward system designed specifically for Neural Fictitious Self-Play (NFSP) convergence:

1. **Soft Penalty (Human-Readable Base Score)**:
   In `judgement/game.py`, raw scores are calculated using the standard Oh Hell rules variant:
   - **Success (`tricks_won == bid`)**: `10 + bid` points
   - **Failure**: `-10 × error` (where error is the absolute difference between bid and tricks won)
   This provides a smooth learning gradient. Missing a bid by 1 trick (-10 points) is immediately recognizable by the neural network as a better outcome than missing it by 5 tricks (-50 points).

2. **Zero-Sum Relative Scaling (RL-Optimized Output)**:
   Competitive self-play algorithms require zero-sum environments to prevent runaway gradients. In `judgement/env.py`, before scores are passed to the NFSP agent, they are converted into a zero-sum relative format:
   - Relative Score = (Agent's Score) - (Average of Opponents' Scores)
   - The result is division-scaled down to keep neural network inputs manageable.
   This mathematically guarantees an agent only receives a positive reward if it actively outperformed its opponents on that specific hand, drastically speeding up Nash Equilibrium convergence.

### 2. **Self-Play Convergence**

The NFSP algorithm may require significant computational resources and episodes to converge to stable Nash equilibrium strategies. Current evaluation against random agents may not reflect true play strength.

**Mitigation**: Implement agent-vs-agent tournament evaluation to better assess learning progress.

## Testing

The project includes a thorough test suite located in the `pytests/` directory. These tests cover:
- **Game Structure**: Initialization, flow, bidding constraints, and trick resolution.
- **History Tracking**: Verification of the 227-feature observation bits and `played_cards_history`.
- **NFSP Agent**: Initializations, policy consistency, and self-play integration.
- **Step Back**: Recursive state restoration for MCTS/ISMCTS support.

### Run All Tests
```bash
uv run pytest pytests/
```

---

For issues, suggestions, or contributions, please open an issue on GitHub.
