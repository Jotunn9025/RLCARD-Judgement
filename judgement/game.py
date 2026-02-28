from typing import List, Dict, Tuple, Union,Any, Optional,Literal
from .card import JudgementCard
from .player import JudgementPlayer
from .dealer import JudgementDealer
import copy

class JudgementGame:
    """
    Judgement is a Trick-Taking Card game and this is its implementation for RLCARD
    
    Game Flow:
    1. Deal cards to 4 players
    2. Bidding Phase where each player bids how manyy tricks they will win
        -Bidding starts from player after Dealer and ends with Dealer
        -Bidding cannot end with total-bids = num_cards
            --In such a case Dealer cannot bid that option and has to choose another one
    3. Playing Phase where players aim to get win tricks equal to their bid
        - Rules
            --They must follow initial suit unless they dont have a card of that suit
            --Highest trump/card of leading suit wins
    4. Scoring:
        Win: +((bid+1)*10+bid)
        Lose:-((bid+1)*10+bid) 
    """

    NUM_PLAYERS=4
    NUM_ACTIONS=66 #14 bids+52 cards

    def __init__(self,allow_step_back:bool = True,starting_set_cards:int=13 ):
        #for things like MCTS
        self.allow_step_back=allow_step_back
        
        #Components of game
        self.dealer = JudgementDealer()
        self.players:List[JudgementPlayer] = [JudgementPlayer(i) for i in range(self.NUM_PLAYERS)]
        
        #Structure
        self.starting_set_cards:int=starting_set_cards
        self.current_set_start:int=starting_set_cards
        self.num_cards:int=starting_set_cards
        self.round_number=1
        self.trump_suit:Literal['S','D','H','C']='S'
        self.dealer_id:int=0
        self.hands:List[List[JudgementCard]]=[[]for _ in range(self.NUM_PLAYERS)]
        #Tracking Phase
        self.phase:Literal['bidding','playing']='bidding'
        self.current_player_id:int=0

        #Bidding Related Stuff
        self.bids:List[Optional[int]]=[None]*self.NUM_PLAYERS
        self.bidding_order:List[int]=[]
        self.bids_made:int=0
        
        #Platying phase variables
        self.tricks_won:List[int]=[]*self.NUM_PLAYERS
        self.current_trick:List[Tuple[int,JudgementCard]]=[]
        self.lead_suit:Optional[Literal['S','D','H','C']]=None
        self.trick_number:int=0
        
        #Scorestory as a dict
        self.cumulative_scores:List[int]=[0]*self.NUM_PLAYERS
        #Game over  flag
        self._game_over:bool=False


        #History for tree search. storing hi
        self.history:List[Dict]=[]
        
        # History of completed tricks for observation
        self.played_cards_history: List[Dict] = []
        
    def init_game(self)->Tuple[ Dict,int]:
        """
        Starts a game and resets everything
        returns a tuple of initial state and first player id
        """
        #RESET
        self.current_set_cards=self.starting_set_cards
        self.num_cards
        self.round_number=1
        self.dealer_id=0
        self.cumulative_scores=[0]*4
        self._game_over=False
        self.history=[]

        return self._init_round()
    
    def _init_round(self)->Tuple[Dict,int]:
        """
        Starts a new round of the set
        Returns a Tuple of initial start and first player id 
        """
        for player in self.players:
            player.reset()

        #Dealing cards and other setup
        self.dealer.create_deck()
        self.dealer.shuffle()
        for player in self.players:
            self.dealer.deal_cards(player,self.num_cards)
        self.hands=[player.hand for player in self.players]
        self.trump_suit=JudgementDealer.get_trump(self.round_number)

        #Bidding phase related code
        self.phase='bidding'
        self.bids=[None]*4
        self.bids_made=0
        self.bidding_order=[(self.dealer_id+i+1)%self.NUM_PLAYERS for i in range(self.NUM_PLAYERS)]
        self.current_player_id=self.bidding_order[0]
        #resetting before bidding
        self.tricks_won=[0]*4
        self.current_trick=[]
        self.lead_suit=None
        self.trick_number=0
        self.played_cards_history = []

        state=self.get_state(self.current_player_id)
        return state,self.current_player_id
    
    def step(self,action:Union[int,JudgementCard])->Tuple[Dict,int]:
        """
        Executes action and makes game move forward
        returns Tuple(next_state,next_player_id)
        """
        if self.allow_step_back:
            self.history.append(self._snapshot())

        if self.phase=='bidding':
            self._process_bid(action)
        else:
            self._process_play(action)
        state=self.get_state(self.current_player_id)
        return state,self.current_player_id
    
    def _snapshot(self)->Dict:
        """
        Get a copy of literally everything in game state so we can step back in algos like MCTS
        """
        return {
            'phase': self.phase,
            'current_player_id': self.current_player_id,
            'bids': self.bids.copy(),
            'bids_made': self.bids_made,
            'bidding_order': self.bidding_order.copy(),
            'tricks_won': self.tricks_won.copy(),
            'current_trick': copy.deepcopy(self.current_trick),
            'lead_suit': self.lead_suit,
            'trick_number': self.trick_number,
            'num_cards': self.num_cards,
            'round_number': self.round_number,
            'current_set_start': self.current_set_start,
            'dealer_id': self.dealer_id,
            'trump_suit': self.trump_suit,
            'cumulative_scores': self.cumulative_scores.copy(),
            '_game_over': self._game_over,
            'players': [
                {
                    'hand': [copy.copy(c) for c in p.hand],
                    'bid': p.bid,
                    'tricks_won': p.tricks_won,
                }
                for p in self.players
            ],
            'played_cards_history': copy.deepcopy(self.played_cards_history),
        }
    def _restore(self,snapshot:Dict):
        """Restore game state using snapshot"""
        self.phase = snapshot['phase']
        self.current_player_id = snapshot['current_player_id']
        self.bids = snapshot['bids']
        self.bids_made = snapshot['bids_made']
        self.bidding_order = snapshot['bidding_order']
        self.tricks_won = snapshot['tricks_won']
        self.current_trick = snapshot['current_trick']
        self.lead_suit = snapshot['lead_suit']
        self.trick_number = snapshot['trick_number']
        self.num_cards = snapshot['num_cards']
        self.round_number = snapshot['round_number']
        self.current_set_start = snapshot['current_set_start']
        self.dealer_id = snapshot['dealer_id']
        self.trump_suit = snapshot['trump_suit']
        self.cumulative_scores = snapshot['cumulative_scores']
        self._game_over = snapshot['_game_over']
        for i, p_snap in enumerate(snapshot['players']):
            self.players[i].hand = p_snap['hand']
            self.players[i].bid = p_snap['bid']
            self.players[i].tricks_won = p_snap['tricks_won']
        self.hands = [p.hand for p in self.players]
        self.played_cards_history = snapshot.get('played_cards_history', [])

    def step_back(self)->bool:
        """
        Revert gamestate to previous i.e. before step call
        returns true if sucessfull
        """
        if not self.history:
            return False
        
        self._restore(self.history.pop())
        return True
    
    def _process_bid(self,bid:int):
        """process bid"""
        player_id=self.current_player_id
        self.bids[player_id]=bid
        self.players[player_id].bid=bid
        self.bids_made+=1
        
        #check if done bidding
        if self.bids_made== self.NUM_PLAYERS:
            self._start_playing_phase()
        else:
            self.current_player_id = self.bidding_order[self.bids_made]
    def _start_playing_phase(self):
        """Switch phase to playing from bidding"""
        self.phase='playing'
        self.trick_number=1
        self.current_player_id=(self.dealer_id+1)%self.NUM_PLAYERS
        self.current_trick=[]
        self.lead_suit=None
    
    def _process_play(self,card:JudgementCard):
        """Process cards"""
        player_id= self.current_player_id
        player=self.players[player_id]
        player.play_card(card)#remove from hand
        self.current_trick.append((player_id,card))
        #first player sets lead suit
        self.lead_suit=card.suit if len(self.current_trick)==1 else self.lead_suit
        #check if done
        if len(self.current_trick)==self.NUM_PLAYERS:
            self._resolve_trick()
        else:
            self.current_player_id=(player_id+1)%self.NUM_PLAYERS

    def _resolve_trick(self):
        """decid trick winner and update state"""
        winner_id=self._determine_winner()
        self.tricks_won[winner_id]+=1
        self.players[winner_id].tricks_won+=1
        
        # Record completed trick
        self.played_cards_history.append({
            'winner_id': winner_id,
            'cards': self.current_trick.copy()
        })
        
        self.current_trick=[]
        self.lead_suit=None
        self.trick_number+=1
        #check if round is done
        if self.phase=='playing' and self.trick_number>self.num_cards:
            round_payoffs=self._calculate_round_payoffs()
            for i in range(self.NUM_PLAYERS):
                self.cumulative_scores[i]+=round_payoffs[i]
            self._advance_round()
        else:
            self.current_player_id=winner_id#winner resumes play as lead

    def _determine_winner(self)->int:
        """Determine winne of current trick. Return winners player id"""
        best_player = self.current_trick[0][0]
        best_card=self.current_trick[0][1]
        for player_id,card in self.current_trick[1:]:
            if self._card_beats(card,best_card):
                best_player=player_id
                best_card=card
        return best_player
    def _card_beats(self,card1:JudgementCard,card2:JudgementCard)->bool:
        """
        Check if card1 beats card2 and returns a bool
        """
        card1_isTrump=card1.suit==self.trump_suit
        card2_isTrump=card2.suit==self.trump_suit
        if card1_isTrump and not card2_isTrump:
            return True
        if not card1_isTrump and card2_isTrump:
            return False
        if card1_isTrump and card2_isTrump:
            return card1.get_rank()>card2.get_rank()
        if card1.suit==self.lead_suit:
            return card1.get_rank()>card2.get_rank()
        return False
    
    def get_state(self,player_id:int)->Dict:
        """Get Game State in Player POV. returns Dict"""
        return{
            'player_id': player_id,
            'hand': self.players[player_id].hand.copy(),
            'phase': self.phase,
            'trump_suit': self.trump_suit,
            'bids': self.bids.copy(),
            'tricks_won': self.tricks_won.copy(),
            'current_trick': self.current_trick.copy(),
            'lead_suit': self.lead_suit,
            'dealer_id': self.dealer_id,
            'num_cards': self.num_cards,
            'played_cards_history': self.played_cards_history.copy(),
            'legal_actions': self.get_legal_actions(player_id)
        }
    
    def get_legal_actions(self,player_id:int=None)->List[int]:
        """
        Get Legal Actions for current player
        Actions: 0-13(Bids), 14-65(Play a card)
        Retuns list of legal action id
        """
        if player_id is None:
            player_id=self.current_player_id
        player=self.players[player_id]
        legal_actions=[]
        if self.phase=='bidding':
            for bid in range(self.num_cards+1):
                if self._check_dealer_bid_legality(player_id,bid):
                    legal_actions.append(bid)
        else:
            playable_cards=self._get_playable_cards(player)
            for card in playable_cards:
                legal_actions.append(14+card.get_index())
        return legal_actions
    
    def _check_dealer_bid_legality(self,player_id:int,bid:int)->bool:
        if player_id==self.dealer_id:
            current_sum=sum(b for b in self.bids if b is not None)
            if current_sum+bid==self.num_cards:
                return False
        return True
    
    def _get_playable_cards(self,player:JudgementPlayer)->List[JudgementCard]:
        """Gets Legal cards player can use"""
        if not self.current_trick or self.lead_suit is None:
            return player.hand.copy()
        card_leadsuit=player.get_cards_of_suit(self.lead_suit)
        if card_leadsuit:
            return card_leadsuit
        return player.hand.copy()
    
    def _end_round(self):
        """Handle end of round chores """
        round_payoffs=self._calculate_round_payoffs()
        for i in range(self.NUM_PLAYERS):
            self.cumulative_scores[i]+=round_payoffs[i]
        self._advance_round()

    def _calculate_round_payoffs(self)->List[int]:
        payoffs = []
        for player in self.players:
            if player.tricks_won == player.bid:
                payoffs.append((player.bid + 1) * 10 + player.bid)
            else:
                payoffs.append(-((player.bid + 1) * 10 + player.bid))
        return payoffs
    
    def _advance_round(self):
        """
        Adnvance to next round or set
        Between sets (rotate dealer and start new set at n-1 cards)
        within sets decrement num cards
        """
        self.round_number+=1
        if self.num_cards>1:
            self.num_cards-=1
            self._init_round()
        else:
            if self.current_set_start==1:
                self._game_over=True
            else:
                self.current_set_start-=1
                self.num_cards=self.current_set_start
                self.dealer_id=(self.dealer_id+1)%self.NUM_PLAYERS
                self._init_round()

    def configure(self,config:Dict):
        """
        Configure Game Parameters
        Settings:
            - 'starting_set_cards': Used to set initial number of cards(default=13)
        """
        if 'starting_set_cards' in config:
            self.starting_set_cards=config['starting_set_cards']
            self.current_set_start=config['starting_set_cards']
            self.num_cards=config['starting_set_cards']
   # Stuff the tests might require
    def is_round_over(self) -> bool:
        if self.phase == 'bidding':
            return False
        return self.trick_number > self.num_cards
    
    def is_over(self) -> bool:
        return self._game_over
    
    def get_num_players(self) -> int:
        return self.NUM_PLAYERS
    
    @staticmethod
    def get_num_actions() -> int:
        return JudgementGame.NUM_ACTIONS
    
    def get_player_id(self) -> int:
        return self.current_player_id
    
    def get_payoffs(self) -> List[float]:
        return self.cumulative_scores.copy()