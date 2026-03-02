import pytest
import torch
import numpy as np
import os
import shutil

from rlcard.agents.nfsp_agent import NFSPAgent
from judgement.env import JudgementEnv

@pytest.fixture
def env():
    return JudgementEnv(config={'starting_set_cards': 3})

@pytest.fixture
def agent(env):
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=torch.device('cpu')
    )

def test_nfsp_initialization(agent, env):
    """Verify detailed initialization state."""
    assert agent is not None
    assert agent._num_actions == env.num_actions
    assert agent._state_shape == env.state_shape[0]
    assert agent.total_t == 0
    assert agent.train_t == 0

def test_nfsp_action_selection_logic(agent, env):
    """Test that mode sampling and action selection work as expected."""
    state, _ = env.reset()
    
    # Force mode: best_response
    agent._mode = 'best_response'
    action = agent.step(state)
    assert action in state['legal_actions']
    
    # Force mode: average_policy
    agent._mode = 'average_policy'
    action = agent.step(state)
    assert action in state['legal_actions']
    
    # Eval step always uses evaluate_with mode
    agent.evaluate_with = 'average_policy'
    action, info = agent.eval_step(state)
    assert action in state['legal_actions']
    assert 'probs' in info

def test_nfsp_training_integration(agent, env):
    """Verify that feeding transitions works and increments counters."""
    state, _ = env.reset()
    action = agent.step(state)
    next_state, next_player_id = env.step(action)
    
    # Dummy transition
    transition = (state, action, 1.0, next_state, False)
    
    initial_total_t = agent.total_t
    agent.feed(transition)
    
    assert agent.total_t == initial_total_t + 1

def test_nfsp_checkpoint_save_restore(agent, env):
    """Verify that saving and restoring from checkpoint preserves state."""
    save_dir = 'pytests/tmp_nfsp'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    try:
        agent.total_t = 1234
        agent.train_t = 56
        filename = 'test_checkpoint.pt'
        
        agent.save_checkpoint(save_dir, filename=filename)
        checkpoint_path = os.path.join(save_dir, filename)
        assert os.path.exists(checkpoint_path)
        
        # Restore
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        new_agent = NFSPAgent.from_checkpoint(checkpoint)
        
        assert new_agent.total_t == 1234
        assert new_agent.train_t == 56
        assert new_agent._num_actions == agent._num_actions
        
    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

def test_nfsp_full_loop_stability():
    """Run a small self-play loop to ensure no crashes over mid-length."""
    env = JudgementEnv(config={'starting_set_cards': 1})
    agents = [
        NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[i],
            hidden_layers_sizes=[16, 16],
            q_mlp_layers=[16, 16],
            device=torch.device('cpu')
        ) for i in range(env.num_players)
    ]
    env.set_agents(agents)
    
    # Run 10 episodes
    for _ in range(10):
        trajectories, payoffs = env.run(is_training=True)
        assert len(trajectories) == 4
        assert len(payoffs) == 4
