import numpy as np
from blackjack_env import BlackjackEnv
import time

# Actions: 0=stand, 1=hit, 2=double, 3=split, 4=insurance

def random_policy(observation):
    # picks a random action every time (hit, stand, double, split, or insurance)
    return np.random.randint(0, 5)


def simple_policy(observation):
    # if player sum is less than 17, we'll hit (same as dealer strategy)
    player_sum, dealer_showing, usable_ace = observation
    
    if player_sum < 17:
        return 1
    else:
        return 0


def basic_strategy_policy(observation):
    # simple basic strategy that considers the usable ace
    player_sum, dealer_showing, usable_ace = observation
    
    if usable_ace:
        if player_sum <= 17:
            return 1
        else:
            return 0
    else:
        if player_sum <= 11:
            return 1
        elif player_sum <= 16:
            if dealer_showing >= 7 or dealer_showing == 1:
                return 1
            else:
                return 0
        else:
            return 0


def human_policy(observation, first_action=True, can_split=False, is_split=False, current_hand=None, insurance_offered=False):
    # manual input policy for human player
    player_sum, dealer_showing, usable_ace = observation
    
    print(f"\nYour hand sum: {player_sum}")
    if current_hand:
        print(f"Current hand: {current_hand}")
    print(f"Dealer showing: {dealer_showing}")
    print(f"Usable ace: {'Yes' if usable_ace else 'No'}")
    print("\nChoose action:")
    print("  0 = STAND")
    print("  1 = HIT")
    
    # Determine valid actions - can double/split on first action of any hand (including split hands)
    if first_action:
        print("  2 = DOUBLE DOWN")
        if can_split:
            print("  3 = SPLIT")
        if insurance_offered:
            print("  4 = INSURANCE (costs 0.5, pays 2:1 if dealer has blackjack)")
        
        valid_actions = [0, 1, 2]
        if can_split:
            valid_actions.append(3)
        if insurance_offered:
            valid_actions.append(4)
        
        # Build action string
        action_parts = ["0", "1", "2"]
        if can_split:
            action_parts.append("3")
        if insurance_offered:
            action_parts.append("4")
        action_str = ", ".join(action_parts[:-1]) + ", or " + action_parts[-1] if len(action_parts) > 1 else action_parts[0]
    else:
        valid_actions = [0, 1]
        action_str = "0 or 1"
    
    while True:
        try:
            action = int(input(f"Enter action ({action_str}): ").strip())
            if action in valid_actions:
                return action
            else:
                print(f"Invalid input. Please enter {action_str}.")
        except (ValueError, KeyboardInterrupt):
            print(f"\nInvalid input. Please enter {action_str}.")


def run_single_episode(env, policy, render=False):
    observation, info = env.reset()
    terminated = False
    total_reward = 0
    steps = 0
    
    while not terminated:
        action = policy(observation)
        observation, reward, terminated, info = env.step(action)
        total_reward += reward
        steps += 1
    
    return total_reward, steps


def evaluate_policy(env, policy, num_episodes=1000, policy_name="Policy"):
    print(f"\n{'='*60}")
    print(f"Evaluating {policy_name} over {num_episodes} episodes")
    print(f"{'='*60}")
    
    rewards = []
    steps_list = []
    wins = 0
    losses = 0
    ties = 0
    blackjacks = 0
    doubles = 0
    splits = 0
    
    for episode in range(num_episodes):
        try:
            total_reward, steps = run_single_episode(env, policy, render=False)
            rewards.append(total_reward)
            steps_list.append(steps)
            
            # Track blackjacks
            if total_reward == 1.5:
                blackjacks += 1
                wins += 1
            # Track splits (split hands can have combined rewards from -2 to +2)
            # We identify splits by checking if the environment has is_split flag
            # For simplicity, count any reward in range as potential split
            # Actual split tracking would need the info dict
            elif total_reward == 1.5:
                blackjacks += 1
                wins += 1
            # Track doubles (rewards are doubled: -2, 0, +2, +3)
            elif total_reward == 3.0 or (abs(total_reward) == 2.0):
                # Could be double or split, but we'll count based on magnitude
                if total_reward > 1.5:
                    doubles += 1
                    wins += 1
                elif total_reward < -1.5:
                    doubles += 1
                    losses += 1
                elif total_reward == 2.0:
                    doubles += 1
                    wins += 1
                elif total_reward == -2.0:
                    doubles += 1
                    losses += 1
                else:
                    ties += 1
            # Regular outcomes
            elif total_reward > 0:
                wins += 1
            elif total_reward == 0:
                ties += 1
            else:
                losses += 1
        except ValueError as e:
            # Handle cases where policy tries invalid actions (e.g., double/split after first action)
            # Count as loss and continue
            losses += 1
            rewards.append(-1.0)
            steps_list.append(1)
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps_list)
    win_rate = (wins / num_episodes) * 100
    loss_rate = (losses / num_episodes) * 100
    tie_rate = (ties / num_episodes) * 100
    blackjack_rate = (blackjacks / num_episodes) * 100
    double_rate = (doubles / num_episodes) * 100
    split_rate = (splits / num_episodes) * 100
    
    print(f"\nResults:")
    print(f"  Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  Average Steps per Episode: {avg_steps:.2f}")
    print(f"\nOutcome Distribution:")
    print(f"  Wins:       {wins:4d} ({win_rate:5.2f}%)")
    print(f"  Losses:     {losses:4d} ({loss_rate:5.2f}%)")
    print(f"  Ties:       {ties:4d} ({tie_rate:5.2f}%)")
    print(f"\nSpecial Actions:")
    print(f"  Blackjacks: {blackjacks:4d} ({blackjack_rate:5.2f}%)")
    print(f"  Doubles:    {doubles:4d} ({double_rate:5.2f}%)")
    print(f"  Splits:     {splits:4d} ({split_rate:5.2f}%)")
    print(f"{'='*60}\n")
    
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "win_rate": win_rate,
        "avg_steps": avg_steps
    }


def run_manual_mode():
    # run in manual/interactive mode where human can input actions
    print("\n" + "="*60)
    print("MANUAL INTERACTIVE MODE")
    print("="*60)
    print("\nYou are playing Blackjack!")
    print("\nActions: 0 = STAND, 1 = HIT, 2 = DOUBLE DOWN, 3 = SPLIT, 4 = INSURANCE")
    print("Note: You can double and split on split hands!")
    print("Note: Unlimited re-splits allowed!")
    print("Note: Insurance is offered when dealer shows an Ace!")
    print("="*60 + "\n")
    
    env = BlackjackEnv(render_mode="human")
    
    play_again = True
    game_num = 0
    session_results = []
    
    while play_again:
        game_num += 1
        print(f"\n{'='*60}")
        print(f"GAME #{game_num}")
        print(f"{'='*60}")
        
        observation, info = env.reset()
        terminated = False
        total_reward = 0
        first_action = True
        can_split = info.get("can_split", False)
        insurance_offered = info.get("insurance_offered", False)
        is_split = False
        
        while not terminated:
            # Check current hand for split detection
            current_hand = None
            if hasattr(env, 'player_hand'):
                current_hand = env.player_hand
                # Check if current hand can be split
                if is_split and first_action and len(current_hand) == 2 and current_hand[0] == current_hand[1]:
                    can_split = True
                elif not first_action:
                    can_split = False
            
            action = human_policy(observation, first_action=first_action, can_split=can_split, 
                                is_split=is_split, current_hand=current_hand, 
                                insurance_offered=insurance_offered)
            action_names = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT", 4: "INSURANCE"}
            action_name = action_names.get(action, "UNKNOWN")
            print(f"\nYou chose: {action_name}")
            
            observation, reward, terminated, info = env.step(action)
            total_reward += reward
            
            # Update state tracking based on action and info
            if action == 4:  # Insurance
                # Check if game ended (dealer has blackjack)
                if not terminated:
                    print("\n[INFO] Dealer doesn't have blackjack. Insurance lost. Continue playing...")
                # Insurance can't be taken again
                insurance_offered = False
            elif action == 3:  # Split or re-split
                is_split = True
                first_action = True  # Reset for next split hand
                can_split = False
                insurance_offered = False
            elif action == 0:  # Stand
                # Check if we're moving to another hand in a split
                if info.get("is_split") and info.get("active_hand", 0) > 0:
                    first_action = True  # Reset for next hand
                    can_split = False
                else:
                    first_action = False
                insurance_offered = False
            elif action in [1, 2]:  # Hit or Double
                first_action = False
                can_split = False
                insurance_offered = False
        
        session_results.append(total_reward)
        
        print("\n" + "-"*60)
        response = input("Play another game? (y/n): ").strip().lower()
        play_again = response in ['y', 'yes']
    
    env.close()
    
    if len(session_results) > 0:
        print("\n" + "="*60)
        print("MANUAL MODE SESSION SUMMARY")
        print("="*60)
        print(f"Games Played: {len(session_results)}")
        print(f"Total Wins: {sum(1 for r in session_results if r > 0)}")
        print(f"Total Losses: {sum(1 for r in session_results if r < 0)}")
        print(f"Total Ties: {sum(1 for r in session_results if r == 0)}")
        print(f"Average Reward: {np.mean(session_results):.4f}")
        print("="*60)

def main():
    print("\n" + "="*60)
    print("BLACKJACK GYMNASIUM ENVIRONMENT DEMONSTRATION")
    print("Group #15 - COMP4010")
    print("="*60)
    
    np.random.seed(42)

    if input("\nRun manual mode? (y/n): ").strip().lower() in ['y', 'yes']:
        run_manual_mode()
    
    input("\nPress Enter to run policy evaluation on random policy")
    
    env = BlackjackEnv()
    
    random_stats = evaluate_policy(env, random_policy, num_episodes=100, policy_name="Random Policy")
    
    input("\nPress Enter to run policy evaluation on simple policy")

    simple_stats = evaluate_policy(env, simple_policy,  num_episodes=100,  policy_name="Simple Policy (Stand on 17+)")
    
    input("\nPress Enter to run policy evaluation on basic strategy policy")
    
    basic_stats = evaluate_policy(env, basic_strategy_policy, num_episodes=100, policy_name="Basic Strategy")
    
    input("\nPress Enter to see policy comparison summary")

    print("\n" + "="*60)
    print("POLICY COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Policy':<30} {'Avg Reward':<15} {'Win Rate':<15}")
    print("-" * 60)
    print(f"{'Random Policy':<30} {random_stats['avg_reward']:>6.4f} ± {random_stats['std_reward']:<5.4f} {random_stats['win_rate']:>6.2f}%")
    print(f"{'Simple Policy':<30} {simple_stats['avg_reward']:>6.4f} ± {simple_stats['std_reward']:<5.4f} {simple_stats['win_rate']:>6.2f}%")
    print(f"{'Basic Strategy Policy':<30} {basic_stats['avg_reward']:>6.4f} ± {basic_stats['std_reward']:<5.4f} {basic_stats['win_rate']:>6.2f}%")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()
