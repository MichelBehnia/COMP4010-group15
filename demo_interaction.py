import numpy as np
from blackjack_env import BlackjackEnv
import time

# 1 is hit, 0 is stand

def random_policy(observation):
    # picks a random action every time (hit or stand)
    return np.random.randint(0, 2)


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


def human_policy(observation):
    # manual input policy for human player
    player_sum, dealer_showing, usable_ace = observation
    
    print(f"\nYour hand sum: {player_sum}")
    print(f"Dealer showing: {dealer_showing}")
    print(f"Usable ace: {'Yes' if usable_ace else 'No'}")
    print("\nChoose action:")
    print("  0 = STAND")
    print("  1 = HIT")
    
    while True:
        try:
            action = int(input("Enter action (0 or 1): ").strip())
            if action in [0, 1]:
                return action
            else:
                print("Invalid input. Please enter 0 or 1.")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please enter 0 or 1.")


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
    
    for episode in range(num_episodes):
        total_reward, steps = run_single_episode(env, policy, render=False)
        rewards.append(total_reward)
        steps_list.append(steps)
        
        if total_reward == 1.5:
            blackjacks += 1
            wins += 1
        elif total_reward > 0:
            wins += 1
        elif total_reward == 0:
            ties += 1
        else:
            losses += 1
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps_list)
    win_rate = (wins / num_episodes) * 100
    loss_rate = (losses / num_episodes) * 100
    tie_rate = (ties / num_episodes) * 100
    blackjack_rate = (blackjacks / num_episodes) * 100
    
    print(f"\nResults:")
    print(f"  Average Steps per Episode: {avg_steps:.2f}")
    print(f"\nOutcome Distribution:")
    print(f"  Wins:       {wins:4d} ({win_rate:5.2f}%)")
    print(f"  Losses:     {losses:4d} ({loss_rate:5.2f}%)")
    print(f"  Ties:       {ties:4d} ({tie_rate:5.2f}%)")
    print(f"  Blackjacks: {blackjacks:4d} ({blackjack_rate:5.2f}%)")
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
    print("\nActions: 0 = STAND, 1 = HIT")
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
        
        while not terminated:
            action = human_policy(observation)
            action_name = "STAND" if action == 0 else "HIT"
            print(f"\nYou chose: {action_name}")
            
            observation, reward, terminated, info = env.step(action)
            total_reward += reward
        
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
    
    random_stats = evaluate_policy(env, random_policy, num_episodes=1000000, policy_name="Random Policy")
    
    input("\nPress Enter to run policy evaluation on simple policy")

    simple_stats = evaluate_policy(env, simple_policy,  num_episodes=1000000,  policy_name="Simple Policy (Stand on 17+)")
    
    input("\nPress Enter to run policy evaluation on basic strategy policy")
    
    basic_stats = evaluate_policy(env, basic_strategy_policy, num_episodes=1000000, policy_name="Basic Strategy")
    
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
