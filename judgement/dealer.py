from typing import List
import numpy as np
from .card import JudgementCard
from .player import JudgementPlayer
import secrets

class JudgementDealer:
    """In charge of deck creation,shuffling, dealing and tracking which suit is trump"""

    TRUMP_ORDER = ['S','D','C','H']

    def __init__(self):
        self.rng=secrets.SystemRandom()
        self.deck:List[JudgementCard]=[]
        # True Random shuffling and deck initialization

    def create_deck(self):
        """Make a completely new deck"""
        self.deck=[]
        for suit in JudgementCard.SUITS:
            for rank in JudgementCard.RANKS:
                self.deck.append(JudgementCard(suit,rank))

    def shuffle(self):
        """Shuffle deck in place"""
        self.rng.shuffle(self.deck)

    def deal_cards(self,player:JudgementPlayer,num_cards:int):
        """Give player specified number of cards"""
        for _ in range(num_cards):
            if self.deck:
                player.hand.append(self.deck.pop())

    @classmethod
    def get_trump(cls,round_number:int)->str:
        """
        Get trump suit for given round number
        Order is as follows
        Spade, Diamonds, Clubs, Hearts
        """
        return cls.TRUMP_ORDER[(round_number-1)%4]