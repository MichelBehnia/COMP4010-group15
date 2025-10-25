import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any


class BlackjackEnv(gym.Env):
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.action_space = spaces.Discrete(2) # since we only have 2 actions for now
        
        # based on the readme
        self.observation_space = spaces.Tuple((
            spaces.Discrete(19, start=4),     
            spaces.Discrete(10, start=1),    
            spaces.Discrete(2)                
        ))
        
        self.render_mode = render_mode
        
        self.player_hand = []
        self.dealer_hand = []
        self.player_sum = 0
        self.dealer_showing = 0
        self.usable_ace = False
        self.natural_blackjack = False  
        
    def _draw_card(self) -> int:
        card = np.random.randint(1, 14)  
        return min(card, 10)  # converts jack, queen, king to 10
    
    def _calculate_hand(self, hand: list) -> Tuple[int, bool]:
        total = sum(hand)
        usable_ace = False
        
        if 1 in hand:
            alt_total = total + 10
            if alt_total <= 21:
                total = alt_total
                usable_ace = True
        
        if total > 21:
            total = 22
            usable_ace = False
        
        return total, usable_ace
    
    def _is_natural_blackjack(self, hand: list) -> bool:
        if len(hand) != 2:
            return False
        total, _ = self._calculate_hand(hand)
        return total == 21
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, int, int], Dict]:
        super().reset(seed=seed)
        
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        
        self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
        
        self.dealer_showing = self.dealer_hand[0]
        
        if hasattr(self, 'dealer_sum'):
            delattr(self, 'dealer_sum')
        
        self.natural_blackjack = self._is_natural_blackjack(self.player_hand)
        
        observation = (self.player_sum, self.dealer_showing, int(self.usable_ace))
        info = {"natural_blackjack": self.natural_blackjack}
        
        if self.render_mode == "human":
            self._render_state()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict]:
        # we assume action input is always valid

        terminated = False
        reward = 0.0
        info = {}
        
        # hit case
        if action == 1:  
            card = self._draw_card()
            self.player_hand.append(card)
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
            
            if self.player_sum == 22:  
                reward = -1.0
                terminated = True
                info["outcome"] = "player_bust"
            else:
                reward = 0.0
                
        # stand case
        elif action == 0:  
            terminated = True
            reward = self._play_dealer()
            
        observation = (self.player_sum, self.dealer_showing, int(self.usable_ace))
        
        if self.render_mode == "human":
            self._render_state()
            if terminated:
                self._render_outcome(reward, info)
        
        return observation, reward, terminated, info
    
    def _play_dealer(self) -> float:

        dealer_sum, dealer_usable_ace = self._calculate_hand(self.dealer_hand)
        
        while dealer_sum < 17:
            card = self._draw_card()
            self.dealer_hand.append(card)
            dealer_sum, dealer_usable_ace = self._calculate_hand(self.dealer_hand)
        
        self.dealer_sum = dealer_sum
        
        dealer_natural = self._is_natural_blackjack(self.dealer_hand)
        
        #reward cases
        if dealer_sum == 22:
            return 1.0  
        
        if self.natural_blackjack and not dealer_natural:
            return 1.5  
        
        if self.player_sum > dealer_sum:
            return 1.0  
        elif self.player_sum == dealer_sum:
            return 0.0  
        else:
            return -1.0  
    
    def _render_state(self):
        print("\n" + "=" * 50)
        print(f"Player's Hand: {self.player_hand} -> Sum: {self.player_sum}")
        print(f"Usable Ace: {self.usable_ace}")
        print(f"Dealer Showing: {self.dealer_showing}")
        if len(self.dealer_hand) > 1 and hasattr(self, 'dealer_sum'):
            print(f"Dealer's Hand: {self.dealer_hand} -> Sum: {self.dealer_sum}")
        print("=" * 50)
    
    def _render_outcome(self, reward: float, info: Dict):
        print("\n" + "=" * 50)
        print("GAME OVER")
        if reward == 1.5:
            print("*** BLACKJACK! Player wins with a natural 21! ***")
        elif reward == 1.0:
            print("[WIN] Player wins!")
        elif reward == 0.0:
            print("[TIE] Push (Tie)")
        elif reward == -1.0:
            if info.get("outcome") == "player_bust":
                print("[LOSE] Player busts! Dealer wins.")
            else:
                print("[LOSE] Dealer wins!")
        print(f"Reward: {reward}")
        print("=" * 50 + "\n")
    
    def render(self):
        if self.render_mode == "human":
            self._render_state()
    
    def close(self):
        pass

