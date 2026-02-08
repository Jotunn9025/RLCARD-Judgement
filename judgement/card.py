class JudgmentCard:
    """Each and every card in the game is represented as an instance of this class.
    """
    SUITS=['S','D','H','C']
    RANKS=['2','3','4','5','6','7','8','9','10','J','Q','K','A']


    def __init__(self,suit,rank):
        """Iniitialize card with given suit and rank"""
        self.suit=suit
        self.rank=rank

    def get_index(self)->int:
        """Unique Integer Value Associated with each card"""
        return self.SUITS.index(self.suit)* 13 + self.RANKS.index(self.rank)
    
    def get_rank(self)->int:
        return self.RANKS.index(self.rank)
    
    @classmethod
    def make_from_index(cls,index:int)-> "JudgmentCard":
        """Creates a card from its index value"""
        return cls(cls.SUITS[index//13],cls.RANKS[index%13])
    
    def __str__(self)->str:
        return f"Card({self.suit}, {self.rank})"
    
    def __repr__(self)->str:
        return f"JudgementCard({self.suit}, {self.rank})"
    
    def __eq__(self,other)->bool:
        if isinstance(other,JudgmentCard):
            return self.suit==other.suit and self.rank==other.rank
        return False
    
    def __hash__(self)->int:
        return self.get_index()
