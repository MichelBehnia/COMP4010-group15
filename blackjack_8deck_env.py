import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any


class Blackjack8DeckEnv(gym.Env):
    
    def __init__(self, render_mode: Optional[str] = None, num_decks: int = 8):
        super().__init__()
        
        self.action_space = spaces.Discrete(5)
        
        # Observation space includes true count as 4th element
        self.observation_space = spaces.Tuple((
            spaces.Discrete(19, start=4),     # Player sum (4-22)
            spaces.Discrete(10, start=1),     # Dealer showing (1-10)
            spaces.Discrete(2),               # Usable ace (0 or 1)
            spaces.Discrete(5, start=-2)      # True count (-2 to +2)
        ))
        
        self.render_mode = render_mode
        self.num_decks = num_decks
        
        self.shoe = []
        self.shuffle_threshold = 0.25  # Reshuffle when 25% or less cards remain
        
        self.running_count = 0
        self.cards_seen = 0
        
        self._initialize_shoe()
        
        self.player_hand = []
        self.dealer_hand = []
        self.player_sum = 0
        self.dealer_showing = 0
        self.usable_ace = False
        self.natural_blackjack = False
        self.first_action = True
        
        self.insurance_taken = False
        self.insurance_offered = False
        
        self.is_split = False
        self.split_hands = []
        self.active_hand_index = 0
        self.hand_first_actions = []
        self.hand_doubled = []
        self.hand_complete = []
    
    def _initialize_shoe(self):
        # Create 8-deck shoe: each deck has 4 of each rank (1-13)
        self.shoe = []
        for _ in range(self.num_decks):
            for rank in range(1, 14):
                for _ in range(4):
                    card_value = min(rank, 10)  # J, Q, K all become 10
                    self.shoe.append(card_value)
        
        np.random.shuffle(self.shoe)
        
        self.running_count = 0
        self.cards_seen = 0
    
    def _needs_reshuffle(self) -> bool:
        total_cards = self.num_decks * 52
        remaining = len(self.shoe)
        return remaining <= (total_cards * self.shuffle_threshold)
    
    def _draw_card(self) -> int:
        if len(self.shoe) == 0 or self._needs_reshuffle():
            self._initialize_shoe()
        
        card = self.shoe.pop()
        
        self._update_count(card)
        
        return card
    
    def _update_count(self, card: int):
        # Hi-Lo card counting system
        # Low cards (2-6): +1
        # Neutral (7-9): 0
        # High cards (10, A): -1
        if 2 <= card <= 6:
            self.running_count += 1
        elif card == 10 or card == 1:
            self.running_count -= 1
        self.cards_seen += 1
    
    def _get_true_count(self) -> int:
        cards_remaining = len(self.shoe)
        decks_remaining = max(1.0, cards_remaining / 52.0)
        true_count = self.running_count / decks_remaining
        
        if true_count <= -2.0:
            return -2
        elif true_count <= -0.5:
            return -1
        elif true_count <= 0.5:
            return 0
        elif true_count <= 2.0:
            return 1
        else:
            return 2
    
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Tuple[int, int, int, int], Dict]:
        super().reset(seed=seed, options=options)
        
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        
        self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
        
        self.dealer_showing = self.dealer_hand[0]
        
        if hasattr(self, 'dealer_sum'):
            delattr(self, 'dealer_sum')
        
        self.natural_blackjack = self._is_natural_blackjack(self.player_hand)
        self.first_action = True
        
        self.insurance_taken = False
        self.insurance_offered = (self.dealer_showing == 1)
        
        self.is_split = False
        self.split_hands = []
        self.active_hand_index = 0
        self.hand_first_actions = []
        self.hand_doubled = []
        self.hand_complete = []
        
        true_count = self._get_true_count()
        
        observation = (self.player_sum, self.dealer_showing, int(self.usable_ace), true_count)
        can_split = len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]
        info = {
            "natural_blackjack": self.natural_blackjack, 
            "can_split": can_split,
            "insurance_offered": self.insurance_offered,
            "cards_remaining": len(self.shoe),
            "running_count": self.running_count,
            "true_count": true_count
        }
        
        if self.render_mode == "human":
            self._render_state()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, bool, Dict]:
        # we assume action input is always valid

        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        # Handle split hands
        if self.is_split:
            return self._step_split_hand(action)
                
        # stand case
        if action == 0:  
            terminated = True
            reward = self._play_dealer()

        # hit case
        elif action == 1:  
            card = self._draw_card()
            self.player_hand.append(card)
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
            
            if self.player_sum == 22:  
                reward = -1.0
                terminated = True
                info["outcome"] = "player_bust"
            else:
                reward = 0.0
            
            self.first_action = False
            
        # double down case
        elif action == 2:
            # Draw one card and stand
            card = self._draw_card()
            self.player_hand.append(card)
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
            
            terminated = True
            
            if self.player_sum == 22:  
                # Player busts lose double bet
                reward = -2.0
                info["outcome"] = "player_bust_double"
            else:
                # Play dealer and double the reward
                base_reward = self._play_dealer()
                reward = base_reward * 2.0
                info["outcome"] = "double_down"
            
            self.first_action = False
        
        # split case
        elif action == 3:
            # Check if splitting aces (value == 1)
            splitting_aces = (self.player_hand[0] == 1)
            
            # Split the hand create two hands
            self.is_split = True
            hand_1 = [self.player_hand[0], self._draw_card()]
            hand_2 = [self.player_hand[1], self._draw_card()]
            
            # Initialize split hands array
            self.split_hands = [hand_1, hand_2]
            self.hand_first_actions = [True, True]
            self.hand_doubled = [False, False]
            self.hand_complete = [False, False]
            
            # If splitting aces automatically stand on both hands (no further actions allowed)
            if splitting_aces:
                # Mark both hands as complete immediately no further actions allowed
                self.hand_complete = [True, True]
                # Play dealer and evaluate all hands
                terminated = True
                reward = self._evaluate_split_hands()
                info["outcome"] = "split_aces_complete"
            else:
                # Start playing hand 0 (first hand) normally for non-ace splits
                self.active_hand_index = 0
                self.player_hand = self.split_hands[0]
                self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
                info["outcome"] = "split"
                info["active_hand"] = 0
            
            self.first_action = False
        
        # insurance case
        elif action == 4:
            # Take insurance (costs 0.5)
            self.insurance_taken = True
            
            # Check if dealer has blackjack immediately
            dealer_natural = self._is_natural_blackjack(self.dealer_hand)
            
            if dealer_natural:
                terminated = True
                
                self.dealer_sum, _ = self._calculate_hand(self.dealer_hand)
                
                insurance_payout = 0.5
                
                # Check if player also has blackjack
                if self.natural_blackjack:
                    # Both have blackjack push on main bet, insurance wins
                    main_bet_result = 0.0
                    reward = main_bet_result + insurance_payout
                    info["outcome"] = "dealer_blackjack_insurance_push"
                else:
                    # Dealer wins main bet so insurance wins
                    main_bet_result = -1.0
                    reward = main_bet_result + insurance_payout
                    info["outcome"] = "dealer_blackjack_insurance_win"
            else:
                info["outcome"] = "insurance_lost_continue"
                # Insurance loss will be deducted at end of game
            
        observation = (self.player_sum, self.dealer_showing, int(self.usable_ace), self._get_true_count())
        
        if self.render_mode == "human":
            self._render_state()
            if terminated:
                self._render_outcome(reward, info)
        
        return observation, reward, terminated, truncated, info
    
    def _step_split_hand(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, bool, Dict]:
        terminated = False
        truncated = False
        reward = 0.0
        info = {"is_split": True, "active_hand": self.active_hand_index}
        
        current_hand = self.split_hands[self.active_hand_index]
        is_first_action = self.hand_first_actions[self.active_hand_index]
        
        # Process action for current hand
        if action == 0:  # Stand
            terminated, info = self._move_to_next_hand(info)
            if terminated:
                reward = self._evaluate_split_hands()
                
        elif action == 1:  # Hit
            card = self._draw_card()
            current_hand.append(card)
            
            # Update the hand
            self.split_hands[self.active_hand_index] = current_hand
            self.hand_first_actions[self.active_hand_index] = False
                
            self.player_hand = current_hand
            self.player_sum, self.usable_ace = self._calculate_hand(current_hand)
            
            # Check for bust
            if self.player_sum == 22:
                info["hand_bust"] = self.active_hand_index
                terminated, info = self._move_to_next_hand(info)
                if terminated:
                    reward = self._evaluate_split_hands()
        
        # Double down on split hand
        elif action == 2:
            card = self._draw_card()
            current_hand.append(card)
            
            # Update the hand and mark as doubled
            self.split_hands[self.active_hand_index] = current_hand
            self.hand_doubled[self.active_hand_index] = True
            self.hand_first_actions[self.active_hand_index] = False
            
            self.player_hand = current_hand
            self.player_sum, self.usable_ace = self._calculate_hand(current_hand)
            
            info["hand_doubled"] = self.active_hand_index
            
            # Move to next hand (or finish if this was the last hand)
            terminated, info = self._move_to_next_hand(info)
            if terminated:
                reward = self._evaluate_split_hands()
        
        # Split again (re-split)
        elif action == 3:
            # Re-split the current hand into two hands
            hand_a = [current_hand[0], self._draw_card()]
            hand_b = [current_hand[1], self._draw_card()]
            
            # Replace current hand with first split, insert second split after it
            self.split_hands[self.active_hand_index] = hand_a
            self.split_hands.insert(self.active_hand_index + 1, hand_b)
            
            # Update tracking arrays by inserting new entries for the new hand
            self.hand_first_actions[self.active_hand_index] = True
            self.hand_first_actions.insert(self.active_hand_index + 1, True)
            
            self.hand_doubled[self.active_hand_index] = False
            self.hand_doubled.insert(self.active_hand_index + 1, False)
            
            self.hand_complete[self.active_hand_index] = False
            self.hand_complete.insert(self.active_hand_index + 1, False)
            
            self.player_hand = hand_a
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
            
            info["outcome"] = "re_split"
            info["active_hand"] = self.active_hand_index
            info["num_hands"] = len(self.split_hands)
            
        observation = (self.player_sum, self.dealer_showing, int(self.usable_ace), self._get_true_count())
        
        if self.render_mode == "human":
            self._render_state()
            if terminated:
                self._render_outcome(reward, info)
        
        return observation, reward, terminated, truncated, info
    
    def _move_to_next_hand(self, info: Dict) -> Tuple[bool, Dict]:
        self.hand_complete[self.active_hand_index] = True
        
        self.active_hand_index += 1
        
        # Check if there are more hands to play
        if self.active_hand_index < len(self.split_hands):
            # More hands to play
            self.player_hand = self.split_hands[self.active_hand_index]
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
            info["active_hand"] = self.active_hand_index
            return False, info
        else:
            # All hands complete, evaluate and play dealer
            info["outcome"] = "split_complete"
            return True, info
    
    def _evaluate_split_hands(self) -> float:
        # Play dealer
        dealer_sum, _ = self._calculate_hand(self.dealer_hand)
        while dealer_sum < 17:
            card = self._draw_card()
            self.dealer_hand.append(card)
            dealer_sum, _ = self._calculate_hand(self.dealer_hand)
        
        self.dealer_sum = dealer_sum
        
        total_reward = 0.0
        for i, hand in enumerate(self.split_hands):
            hand_sum, _ = self._calculate_hand(hand)
            
            if hand_sum == 22:
                reward = -1.0
            elif dealer_sum == 22:
                reward = 1.0
            elif hand_sum > dealer_sum:
                reward = 1.0
            elif hand_sum == dealer_sum:
                reward = 0.0
            else:
                reward = -1.0
            
            if self.hand_doubled[i]:
                reward *= 2.0
            
            total_reward += reward
        
        return total_reward
    
    def _play_dealer(self) -> float:

        dealer_sum, dealer_usable_ace = self._calculate_hand(self.dealer_hand)
        
        while dealer_sum < 17:
            card = self._draw_card()
            self.dealer_hand.append(card)
            dealer_sum, dealer_usable_ace = self._calculate_hand(self.dealer_hand)
        
        self.dealer_sum = dealer_sum
        
        dealer_natural = self._is_natural_blackjack(self.dealer_hand)
        
        if dealer_sum == 22:
            base_reward = 1.0  
        elif self.natural_blackjack and not dealer_natural:
            base_reward = 1.5  
        elif self.player_sum > dealer_sum:
            base_reward = 1.0  
        elif self.player_sum == dealer_sum:
            base_reward = 0.0  
        else:
            base_reward = -1.0
        
        insurance_reward = 0.0
        if self.insurance_taken:
            if dealer_natural:
                # Net: +1.0 payout - 0.5 cost = +0.5
                insurance_reward = 0.5
            else:
                insurance_reward = -0.5
        
        return base_reward + insurance_reward  
    
    def _render_state(self):
        print("\n" + "=" * 50)
        if self.is_split:
            print(f"SPLIT HANDS ({len(self.split_hands)} hands):")
            for i, hand in enumerate(self.split_hands):
                hand_sum, _ = self._calculate_hand(hand)
                status = '<-- ACTIVE' if i == self.active_hand_index else '(Complete)' if self.hand_complete[i] else ''
                status += ' DOUBLED' if self.hand_doubled[i] else ''
                print(f"  Hand {i+1}: {hand} -> Sum: {hand_sum} {status}")
        else:
            print(f"Player's Hand: {self.player_hand} -> Sum: {self.player_sum}")
        print(f"Usable Ace: {self.usable_ace}")
        if self.insurance_taken:
            print(f"Insurance: TAKEN")
        print(f"Dealer Showing: {self.dealer_showing}")
        if len(self.dealer_hand) > 1 and hasattr(self, 'dealer_sum'):
            print(f"Dealer's Hand: {self.dealer_hand} -> Sum: {self.dealer_sum}")
        print(f"Cards Remaining in Shoe: {len(self.shoe)}")
        print("=" * 50)
    
    def _render_outcome(self, reward: float, info: Dict):
        print("\n" + "=" * 50)
        print("GAME OVER")
        
        outcome = info.get("outcome", "")
        
        if outcome == "split_aces_complete":
            print(f"[SPLIT ACES] Aces split - one card dealt to each hand (no further actions)")
            hand_1_sum, _ = self._calculate_hand(self.split_hands[0])
            hand_2_sum, _ = self._calculate_hand(self.split_hands[1])
            print(f"  Hand 1: {self.split_hands[0]} -> Sum: {hand_1_sum}")
            print(f"  Hand 2: {self.split_hands[1]} -> Sum: {hand_2_sum}")
            print(f"  Dealer: {self.dealer_hand} -> Sum: {self.dealer_sum}")
            
            if reward == 2.0:
                print("[WIN] Both hands won!")
            elif reward == 1.0:
                print("[MIXED] One hand won, one hand lost/tied")
            elif reward == 0.0:
                print("[TIE] Both hands tied, or one won and one lost")
            elif reward == -1.0:
                print("[MIXED] One hand lost, one hand won/tied")
            elif reward == -2.0:
                print("[LOSE] Both hands lost!")
            else:
                print(f"[RESULT] Net result across both hands")
            print(f"Reward: {reward}")
            print("=" * 50 + "\n")
            return
        
        if outcome == "dealer_blackjack_insurance_push":
            print("[DEALER BLACKJACK] Dealer has blackjack!")
            print("[INSURANCE] Insurance paid out! (Net +0.5)")
            print("[MAIN BET] Both have blackjack - Push (Tie)")
            print(f"Final Result: Insurance wins (+0.5), Main bet ties (0) = {reward}")
            print(f"Reward: {reward}")
            print("=" * 50 + "\n")
            return
        elif outcome == "dealer_blackjack_insurance_win":
            print("[DEALER BLACKJACK] Dealer has blackjack!")
            print("[INSURANCE] Insurance paid out! (Net +0.5)")
            print("[MAIN BET] Lost to dealer blackjack (-1.0)")
            print(f"Final Result: Insurance wins (+0.5), Main bet loses (-1.0) = {reward}")
            print(f"Reward: {reward}")
            print("=" * 50 + "\n")
            return
        
        if self.insurance_taken and hasattr(self, 'dealer_sum'):
            dealer_natural = self._is_natural_blackjack(self.dealer_hand)
            if dealer_natural:
                print("[INSURANCE] Insurance paid out! (Net +0.5)")
            else:
                print("[INSURANCE] Insurance lost. (Net -0.5)")
        
        
        if outcome == "split_complete" or outcome == "re_split":
            num_hands = len(self.split_hands)
            print(f"[SPLIT] All {num_hands} hand(s) played!")
            
            for i, hand in enumerate(self.split_hands):
                hand_sum, _ = self._calculate_hand(hand)
                doubled = " (DOUBLED)" if self.hand_doubled[i] else ""
                print(f"  Hand {i+1}: {hand} -> Sum: {hand_sum}{doubled}")
            
            print(f"  Dealer: {self.dealer_hand} -> Sum: {self.dealer_sum}")
            
            if reward >= num_hands:
                print(f"[WIN] All {num_hands} hands won!")
            elif reward <= -num_hands:
                print(f"[LOSE] All {num_hands} hands lost!")
            elif reward > 0:
                print(f"[WIN] Net positive result across all hands")
            elif reward < 0:
                print(f"[LOSE] Net negative result across all hands")
            else:
                print(f"[TIE] Net even result across all hands")
        elif outcome == "player_bust_double":
            print("[LOSE] Player busts on double down! Lost double bet.")
        elif outcome == "double_down":
            if reward > 0:
                print("[WIN] Double down successful! Won double bet!")
            elif reward == 0:
                print("[TIE] Double down push (Tie)")
            else:
                print("[LOSE] Double down failed. Lost double bet.")
        elif reward == 1.5:
            print("*** BLACKJACK! Player wins with a natural 21! ***")
        elif reward == 2.0 and outcome != "split_complete":
            print("[WIN] Player wins double bet!")
        elif reward == 1.0:
            print("[WIN] Player wins!")
        elif reward == 0.0:
            print("[TIE] Push (Tie)")
        elif reward == -1.0:
            if outcome == "player_bust":
                print("[LOSE] Player busts! Dealer wins.")
            else:
                print("[LOSE] Dealer wins!")
        elif reward == -2.0 and outcome != "split_complete":
            print("[LOSE] Lost double bet!")
        else:
            if reward > 0:
                print(f"[WIN] Player wins!")
            elif reward < 0:
                print(f"[LOSE] Dealer wins!")
            
        print(f"Reward: {reward}")
        print("=" * 50 + "\n")
    
    def render(self):
        if self.render_mode == "human":
            self._render_state()
    
    def close(self):
        pass
    
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(5, dtype=bool)
        
        # Stand (0) and Hit (1) are always valid
        mask[0] = True
        mask[1] = True
        
        # Double down (2) only valid on first action
        if self.first_action and not self.is_split:
            mask[2] = True
        elif self.is_split and self.active_hand_index < len(self.hand_first_actions):
            # For split hands, check if current hand's first action
            mask[2] = self.hand_first_actions[self.active_hand_index]
        
        # Split (3) only valid on first action with matching cards
        if self.first_action and len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]:
            mask[3] = True
        elif self.is_split and self.active_hand_index < len(self.hand_first_actions):
            # For split hands, check if re-split is possible
            if self.hand_first_actions[self.active_hand_index]:
                current_hand = self.split_hands[self.active_hand_index]
                if len(current_hand) == 2 and current_hand[0] == current_hand[1] and current_hand[0] != 1:
                    mask[3] = True
        
        # Insurance (4) only valid when offered and on first action
        if self.first_action and self.insurance_offered and not self.insurance_taken:
            mask[4] = True
        
        return mask
