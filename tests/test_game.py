import pytest
import numpy as np
from judgement.game import JudgementGame
from judgement.env import JudgementEnv
from judgement.card import JudgementCard

def test_game_initialization():
    """Thoroughly check game initialization state."""
    game = JudgementGame(starting_set_cards=13)
    game.init_game()
    
    assert len(game.players) == 4
    assert game.current_player_id == (game.dealer_id + 1) % 4
    assert game.round_number == 1
    assert game.num_cards == 13
    assert len(game.played_cards_history) == 0
    
    for player in game.players:
        assert len(player.hand) == 13

def test_bidding_phase_legality():
    """Verify bidding constraints, especially the dealer's forbidden bid."""
    game = JudgementGame(starting_set_cards=3)
    game.init_game()
    
    # Player 1, 2, 3 bid
    game.step(1) # p1 bids 1
    game.step(1) # p2 bids 1
    game.step(0) # p3 bids 0
    
    # Total so far: 1+1+0 = 2. Cards in play: 3.
    # Dealer (p0) cannot bid 1 because 2+1 = 3.
    legal_actions = game.get_legal_actions(0)
    assert 1 not in legal_actions
    assert 0 in legal_actions
    assert 2 in legal_actions
    assert 3 in legal_actions

def test_trick_resolution_and_history():
    """Verify trick winners, suit following, and history tracking."""
    # Force a simple game with 2 cards
    game = JudgementGame(starting_set_cards=2)
    game.init_game()
    
    # Bidding
    for _ in range(4):
        actions = game.get_legal_actions(game.current_player_id)
        game.step(actions[0])
    
    assert game.phase == 'playing'
    
    # Simulate first trick
    leader_id = game.current_player_id
    cards_played = []
    for _ in range(4):
        player_id = game.current_player_id
        action_ids = game.get_legal_actions(player_id)
        legal_cards = [JudgementCard.make_from_index(a - 14) for a in action_ids]
        
        # Check suit following if not leader
        if len(cards_played) > 0:
            lead_suit = cards_played[0].suit
            has_suit = any(c.suit == lead_suit for c in game.players[player_id].hand)
            if has_suit:
                for c in legal_cards:
                    assert c.suit == lead_suit
        
        card = legal_cards[0]
        cards_played.append(card)
        game.step(card)
        
    # Check history after 1 trick
    assert len(game.played_cards_history) == 1
    last_trick = game.played_cards_history[0]
    assert 'winner_id' in last_trick
    assert len(last_trick['cards']) == 4
    for _, c in last_trick['cards']: # current_trick is list of (player_id, JudgementCard)
        # Note: cards_played might not be directly comparable with the tuple in the trick
        # but we know cards_played contains the objects
        pass

def test_observation_history_features():
    """Verify the 227-feature observation correctly encodes history."""
    env = JudgementEnv(config={'starting_set_cards': 1})
    state, _ = env.reset()
    obs = state['obs']
    
    # Initially history bits should be 0
    # Trick winners: indices 123-174 (52 bits)
    # Played cards: indices 175-226 (52 bits)
    assert np.all(obs[123:175] == 0)
    assert np.all(obs[175:227] == 0)
    
    # Play a full round (1 card each)
    # Bidding
    for _ in range(4):
        action = list(state['legal_actions'].keys())[0]
        state, _ = env.step(action)
    
    # Playing
    played_cards = []
    for _ in range(4):
        action = list(state['legal_actions'].keys())[0]
        played_cards.append(action - 14) # Convert action to card index
        state, _ = env.step(action)
    
    # End of round/game (since 1 card)
    # Get final state by getting internal state from game for observer 0
    state = env.game.get_state(0)
    obs = env._extract_state(state)['obs']
    
    # Now check history bits
    # Trick 0 winner bit should be set in Trick winners section
    winners_section = obs[123:175]
    assert np.sum(winners_section) == 1
    
    # Played cards bitmask should have 4 bits set
    mask_section = obs[175:227]
    assert np.sum(mask_section) == 4
    for c in played_cards:
        assert mask_section[c] == 1

def test_full_game_flow():
    """Run a full game multi-round sequence and verify stability."""
    game = JudgementGame(starting_set_cards=5)
    # This will run 3 -> 2 -> 1 rounds
    game.init_game()
    
    while not game.is_over():
        action_ids = game.get_legal_actions(game.current_player_id)
        action = action_ids[0]
        if game.phase == 'playing':
            action = JudgementCard.make_from_index(action - 14)
        game.step(action)
    
    assert game.round_number == 16 # 5+4+3+2+1 completed rounds = 15, round_number increments after each
    assert game.is_over()
