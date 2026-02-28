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
   - **Success**: Win exactly your bid → `(bid + 1) × 10 + bid` points
   - **Failure**: Win more or fewer tricks than bid → `−((bid + 1) × 10 + bid)` points

### Observation Space (227 Features)

The environment provides a compact state representation for agents:

| Component | Size | Description |
|-----------|------|-------------|
| My Hand | 52 bits | One-hot encoding of cards in hand |
| Trump Suit | 4 bits | One-hot encoding of current trump suit |
| Current Trick | 52 bits | One-hot encoding of cards played this trick |
| Bids | 4 floats | Normalized bids of all 4 players (0-1) |
| Tricks Won | 4 floats | Normalized tricks won so far (0-1) |
| Dealer Position | 4 bits | One-hot encoding of dealer |
| Phase Indicator | 1 bit | 0 = Bidding, 1 = Playing |
| My Bid | 1 float | Normalized personal bid |
| My Tricks Won | 1 float | Normalized personal tricks won |
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

## Known Issues & Limitations

### 1. **Inadequate Reward Signal**

**Problem**: The current payoff function is binary (win or lose the bid) with a linear scaling based on bid size. This creates sparse rewards that don't guide agents effectively toward learning optimal strategies.

**Current Formula**: 
- Success (tricks_won == bid): `(bid + 1) × 10 + bid`
- Failure (tricks_won ≠ bid): `−((bid + 1) × 10 + bid)`

**Limitations**:
- No gradient for near-successes (e.g., bidding 5 but winning 4 or 6 tricks gets same penalty as bidding 0 and winning 13)
- Larger bids receive larger absolute rewards/penalties, creating scaling issues
- No intermediate rewards for progress within an episode

**Suggested Improvements**:

1. **Soft Reward Function**: Replace binary success/failure with a graduated reward:
   ```python
   def soft_payoff(bid, tricks_won):
       error = abs(bid - tricks_won)
       return (bid + 1) * 10 + bid - error * penalty_factor
   ```

2. **Normalize Rewards**: Scale payoffs relative to maximum bid to maintain consistent signal strength:
   ```python
   def normalized_payoff(bid, tricks_won, max_bid=13):
       base = (bid + 1) * 10 + bid
       if tricks_won == bid:
           return base / (max_bid + 1)
       else:
           return -base / (max_bid + 1)
   ```

3. **Add Dense Rewards**: Provide partial credit for progress (e.g., +1 per trick won relative to bid):
   ```python
   reward = (bid + 1) * 10 + bid if tricks_won == bid else -(abs(bid - tricks_won) * penalty_per_trick)
   ```

4. **Adaptive Scaling**: Use curriculum learning to adjust reward scale during training (larger penalties early, smaller later)

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
