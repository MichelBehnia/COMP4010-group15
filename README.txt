Group # 15

Members:
Luka G. 
Michel B.
Ben C.
Sam J.



CUSTOM BLACKJACK GYMNASIUM ENVIRONMENT SPECIFICATION:

    This custom Gym environment implements the classic card game Blackjack
    where a player competes against a dealer to achieve a hand
    value as close to 21 as possible without exceeding it.



STATE SPACE:

The states are represented as a tuple of 3 elements:
    State = (player_sum, dealer_showing, usable_ace)

Components:

1. player_sum (int): The current sum of the player's hand
    (if sum is >21 then player sum is set to 22 (bust) so that its one state)
    possible value: 4-22 (19 values)
   
2. dealer_showing (int): The value of the dealer's face-up card
    possible value: 1-10 (10 values)
   
3. usable_ace (bool): True if the player has an ace counted as 11 without busting
    possible value: T/F (2 values)

Total States: 19 x 10 x 2 = 380



ACTION SPACE:

The actions consists of two standard Blackjack actions:

0 = STAND: End your turn and let the dealer play their hand
1 = HIT: Request another card to be added to your hand

Action Constraints:
- Player can HIT multiple times until they STAND or bust (sum > 21)
- Once player chooses STAND, the episode proceeds to dealer's turn



REWARD STRUCTURE:

End-of-Episode Rewards:
+1.5  : Player gets Blackjack (21 with 2 cards) and dealer doesn't
+1.0  : Player wins (higher sum than dealer without busting, or dealer busts)
 0.0  : Push/Tie (same sum as dealer)
-1.0  : Player loses (busts, or dealer has higher sum/blackjack)

Game Rules:
- If player busts (sum > 21): immediate -1.0 reward, episode terminates
- Dealer must hit on 16 or below, stand on 17 or above (standard rules)
- Natural blackjack (21 with first 2 cards) beats regular 21

Reward Timing:
- Intermediate steps return reward = 0.0
- Final reward given only at episode termination



RESET FUNCTION

Purpose: Initialize a new episode

Process:
1. Draw from deck (infinite deck assumption if we're not doing card counting variant)
2. Deal 2 cards to player
3. Deal 2 cards to dealer (1 face-up, 1 face-down)
4. Calculate initial player sum and check for usable ace
5. Determine dealer's showing card

Returns:
- Initial state (player_sum, dealer_showing, usable_ace)

Edge Cases:
- If player is dealt a natural blackjack (21) -> auto-resolve



STEP FUNCTION

Purpose: Execute one action and transition to next state

Input:
- action (int): The action to take (0=STAND, 1=HIT)

Process:
   If action == HIT (1):
     - Draw one card from the deck and add to player's hand
     - Recalculate player_sum and usable_ace status
     - Check if player busts (sum = 22)
     - If bust: terminated = True, reward = -1.0
     - If not bust: continue episode with new state, reward = 0.0
   
   If action == STAND (0):
     - Player's turn ends
     - Dealer reveals hidden card and plays by fixed strategy:
         Hit on sum <= 16
         Stand on sum >= 17
     - Compare final hand sums to determine winner:
        Player sum > Dealer sum (and <= 21): reward = +1.0
        Player sum = Dealer sum: reward = 0.0 (push)
        Player sum < Dealer sum or Dealer has blackjack: reward = -1.0
        Dealer busts (sum > 21): reward = +1.0
     - Set terminated = True with appropriate reward
     - Special case: Player natural blackjack gives +1.5 (if dealer doesn't)

 Update internal state variables
    Returns:
    - state (tuple): Next state (player_sum, dealer_showing, usable_ace)
    - reward (float): Reward obtained from this transition
    - terminated (bool): Whether episode has ended



Termination Conditions:
- Player busts (sum > 21)
- Player chooses to STAND (dealer then plays out their hand)




POTENTIAL VARIANTS AND EXTENSIONS

This base environment can be extended with additional actions to create a more 
complex Blackjack gameplay. 

Two common variants include: 

SURRENDER - allowing the player to forfeit the hand immediately after the 
initial deal and lose only half their bet (reward = -0.5), 

DOUBLE DOWN - allowing the player to double their bet, receive exactly one more card, and 
then automatically stand, with doubled rewards/losses. 

Both actions are typically restricted to the first decision point before any 
hits are made.

Other possible variant: Card counting, Spanish 21, Bahama Bonus
