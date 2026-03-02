import pytest
import copy
from judgement.game import JudgementGame
from judgement.env import JudgementEnv
from judgement.card import JudgementCard

def _step_game(game: JudgementGame):
    action = game.get_legal_actions(game.current_player_id)[0]
    if game.phase == 'playing':
        action = JudgementCard.make_from_index(action - 14)
    game.step(action)

def test_step_back_initial_state():
    """Verify step_back on initial state does nothing or raises appropriately."""
    game = JudgementGame(allow_step_back=True)
    game.init_game()
    assert not game.step_back()

def test_step_back_bidding_flow():
    """Verify step_back restores player and phase during bidding."""
    game = JudgementGame(allow_step_back=True, starting_set_cards=3)
    game.init_game()
    
    initial_player = game.current_player_id
    _step_game(game)
    assert game.current_player_id != initial_player
    
    assert game.step_back()
    assert game.current_player_id == initial_player
    assert game.phase == 'bidding'

def test_step_back_trick_resolution_history():
    """CRITICAL: Verify step_back correctly rolls back history and trick wins."""
    game = JudgementGame(allow_step_back=True, starting_set_cards=1)
    game.init_game()
    
    # Bidding
    for _ in range(4):
        _step_game(game)
        
    # Before playing
    assert len(game.played_cards_history) == 0
    
    # Record state before last card
    # (3 cards played, about to play the 4th which resolves the trick)
    for _ in range(3):
        _step_game(game)
    
    state_before_resolution = copy.deepcopy(game.get_state(game.current_player_id))
    
    # Play 4th card -> Resolves trick
    _step_game(game)
    assert len(game.played_cards_history) == 1
    
    # Step back
    assert game.step_back()
    assert len(game.played_cards_history) == 0
    
    # Verify tricks_won is rolled back
    for p in game.players:
        assert p.tricks_won == 0

def test_recursive_step_back_to_start():
    """Verify we can step back all the way to the first move."""
    game = JudgementGame(allow_step_back=True, starting_set_cards=2)
    game.init_game()
    
    moves = 0
    # Play some moves
    for _ in range(6):
        _step_game(game)
        moves += 1
        
    for _ in range(moves):
        assert game.step_back()
        
    assert game.round_number == 1
    assert game.phase == 'bidding'
    assert not game.step_back()

def test_env_step_back_integration():
    """Verify Environment wrapper supports step_back correctly."""
    env = JudgementEnv(config={'allow_step_back': True, 'starting_set_cards': 1})
    state, _ = env.reset()
    
    action = list(state['legal_actions'].keys())[0]
    env.step(action)
    
    assert env.step_back()
    # Should be back at reset state
    assert env.game.phase == 'bidding'
    assert env.game.current_player_id == (env.game.dealer_id + 1) % 4

def test_step_back_disabled_raises():
    """Verify that if allow_step_back is False, step_back returns False or does nothing."""
    game = JudgementGame(allow_step_back=False)
    game.init_game()
    _step_game(game)
    assert not game.step_back()
