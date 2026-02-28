from typing import Dict, List, OrderedDict
from collections import OrderedDict as ODict
from rlcard.envs import Env
from .game import JudgementGame
from .card import JudgementCard
import numpy as np
class JudgementEnv(Env):
    """
    RLCard Env for Judgement or Oh Hell.

    State Representation (obs vector):
    - My Hand: 52 bits (one-hot)
    - Trump Suit: 4 bits (one-hot)
    - Current Trick: 52 bits (one-hot cards on table)
    - Bids: 4 floats (normalized 0-1)
    - Tricks Won: 4 floats (normalized 0-1)
    - Dealer Position: 4 bits (one-hot)
    - Phase Indicator: 1 bit (0=bidding, 1=playing)
    - My Bid: 1 float (normalized)
    - My Wins: 1 float (normalized)
    - Trick Winners: 52 bits (13 tricks * 4 players)
    - Played Cards Bitmask: 52 bits
    Total: 227 features

    Action Space (66 actions):
    - 0-13: Bid 0-13 tricks
    - 14-65: Play card (card_index + 14)

    """

    def __init__(self, config:Dict=None):
        
        config=config or {}
        self.name='Judgement'
        self.game = JudgementGame(allow_step_back=config.get('allow_step_back', True))
        self.game.configure(config)
        self.NUM_PLAYERS = self.game.NUM_PLAYERS
        if 'seed' not in config:
            config['seed']=None
        if 'allow_step_back' not in config:
            config['allow_step_back']=True
        super().__init__(config)

        self.state_shape = [[227] for _ in range(self.NUM_PLAYERS)]
        self.action_shape = [None for _ in range(self.NUM_PLAYERS)]

    def _extract_state(self, state:Dict)->Dict:
        """
        Converts game state to rl observation
        """
        player_id=state['player_id']
        obs_parts=[]
        #hand representation
        hand_rep=np.zeros(52,dtype=np.float32)
        for card in state['hand']:
            hand_rep[card.get_index()]=1
        obs_parts.append(hand_rep)
        #trump suit representation
        trump_rep = np.zeros(self.NUM_PLAYERS, dtype=np.float32)
        trump_idx = JudgementCard.SUITS.index(state['trump_suit'])
        trump_rep[trump_idx] = 1
        obs_parts.append(trump_rep)
        # current trick
        trick_rep = np.zeros(52, dtype=np.float32)
        for _, card in state['current_trick']:
            trick_rep[card.get_index()] = 1
        obs_parts.append(trick_rep)
        #bdis
        max_cards = 13
        bids_rep = np.zeros(self.NUM_PLAYERS, dtype=np.float32)
        for i, bid in enumerate(state['bids']):
            if bid is not None:
                bids_rep[i] = bid / max_cards
        obs_parts.append(bids_rep)
        #tricks won
        tricks_rep = np.array(state['tricks_won'], dtype=np.float32) / max_cards
        obs_parts.append(tricks_rep)
        #DEALER POS
        dealer_rep = np.zeros(self.NUM_PLAYERS, dtype=np.float32)
        dealer_rep[state['dealer_id']] = 1
        obs_parts.append(dealer_rep)
        #phase indicatior
        phase_rep = np.array([1.0 if state['phase'] == 'playing' else 0.0], dtype=np.float32)
        obs_parts.append(phase_rep)
        #my bid
        my_bid = state['bids'][player_id]
        my_bid_rep = np.array([my_bid / max_cards if my_bid is not None else 0.0], dtype=np.float32)
        obs_parts.append(my_bid_rep)
        #my wins
        my_wins_rep = np.array([state['tricks_won'][player_id] / max_cards], dtype=np.float32)
        obs_parts.append(my_wins_rep)
        #winners
        winners_rep = np.zeros(52, dtype=np.float32)
        if 'played_cards_history' in state:
            for i, trick in enumerate(state['played_cards_history']):
                if i < 13:
                    winner_id = trick['winner_id']
                    winners_rep[i * self.NUM_PLAYERS + winner_id] = 1
        obs_parts.append(winners_rep)
        #played cards
        played_cards_rep = np.zeros(52, dtype=np.float32)
        if 'played_cards_history' in state:
            for trick in state['played_cards_history']:
                for _, card in trick['cards']:
                    played_cards_rep[card.get_index()] = 1
        obs_parts.append(played_cards_rep)
        

        obs=np.concatenate(obs_parts)
        
        legal_action_ids = state['legal_actions']
        legal_actions = ODict({action_id: None for action_id in legal_action_ids})
        
        return {
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': obs,
            'raw_legal_actions': legal_action_ids
        }

    def _decode_action(self, action_id):
        """Convert action id to game action"""
        if action_id <= 13:
            return action_id
        else:
            card_index = action_id - 14
            return JudgementCard.make_from_index(card_index)
        
    def _get_legal_actions(self) -> List[int]:
        """Get legal action IDs for current player"""
        return self.game.get_legal_actions()
    
    def get_payoffs(self) -> np.ndarray:
        """Get payoffs per player"""
        return np.array(self.game.get_payoffs())
    
    def get_perfect_information(self) -> Dict:
        """
        Get complete game state (for debugging/analysis)."""
        return {
            'phase': self.game.phase,
            'trump_suit': self.game.trump_suit,
            'num_cards': self.game.num_cards,
            'dealer_id': self.game.dealer_id,
            'current_player_id': self.game.current_player_id,
            'bids': self.game.bids.copy(),
            'tricks_won': self.game.tricks_won.copy(),
            'current_trick': [(pid, str(c)) for pid, c in self.game.current_trick],
            'hands': [[str(c) for c in self.game.players[i].hand] for i in range(self.NUM_PLAYERS)]
        }