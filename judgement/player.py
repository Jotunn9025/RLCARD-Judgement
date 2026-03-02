from typing import List
from .card import JudgementCard

class JudgementPlayer:
    def __init__(self,player_id):
        """Make New Player"""
        self.player_id=player_id
        self.hand:List[JudgementCard]=[]
        self.bid:int=None
        self.tricks_won:int=0

    def reset(self):
        """Resets Player"""
        self.hand=[]
        self.bid=None
        self.tricks_won=0

    def get_hand_indices(self)->List[int]:
        """Gets unique card index for each card in hand"""
        return [card.get_index() for card in self.hand]
    
    def has_suit(self,suit:str)->bool:
        """Checks if player has at least one of suit"""
        return any(card.suit==suit for card in self.hand)
    
    def get_cards_of_suit(self,suit:str)->List[JudgementCard]:
        """Gets all cards of suit of the player"""
        return [card for card in self.hand if card.suit==suit]
    
    def play_card(self,card:JudgementCard)->JudgementCard:
        """Play card that is passed as an argument and remove it from hand"""
        self.hand.remove(card)
        return card
    
    def __str__(self) -> str:
        return f"Player {self.player_id}"
    
    def __repr__(self)->str:
        return f"JudgementPlayer({self.player_id})"
    
    def __eq__(self,other)->bool:
        if isinstance(other,JudgementPlayer):
            return self.player_id==other.player_id
        return False
    
    def __hash__(self)->int:
        return self.player_id
