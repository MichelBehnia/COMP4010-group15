import numpy as np
from collections import defaultdict
#from blackjack_env import BlackjackEnv
from blackjack_8deck_env import Blackjack8DeckEnv as BlackjackEnv

NUM_EPISODES = 100000
NUM_EVAL_GAMES = 10000

class MonteCarloBlackjack:
    def __init__(self, episodes=NUM_EPISODES, epsilon=0.05, gamma=1.0):
        self.episodes = episodes
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-values: Q[(state, action)] = value
        self.Q = defaultdict(float)
        
        # Returns: stores returns for each (state, action) pair
        self.returns = defaultdict(list)
        
        # Action tracking
        self.action_counts = {
            0: 0,  # stand
            1: 0,  # hit
            2: 0,  # double
            3: 0,  # split
            4: 0   # insurance
        }
        
        self.action_names = {
            0: "Stand",
            1: "Hit",
            2: "Double",
            3: "Split",
            4: "Insurance"
        }
        
        # Game outcome tracking
        self.total_return = 0.0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.episode_returns = []
        
    def get_valid_actions(self, env):
        # Get list of valid actions from environment action mask
        action_mask = env.action_masks()
        valid_actions = [i for i in range(len(action_mask)) if action_mask[i]]
        return valid_actions
    
    def choose_action(self, state, valid_actions):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration: choose random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploitation: choose best valid action
            q_values = [self.Q[(state, a)] for a in valid_actions]
            max_q = max(q_values)
            # If multiple actions have same Q-value, choose randomly among them
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def train(self):
        env = BlackjackEnv()
        
        print(f"Training Monte Carlo agent for {self.episodes} episodes...")
        print("=" * 70)
        
        for episode in range(self.episodes):
            episode_data = []
            obs, info = env.reset()
            state = obs
            terminated = False
            first_action = True
            
            while not terminated:
                valid_actions = self.get_valid_actions(env)
                action = self.choose_action(state, valid_actions)
                
                # Track action
                self.action_counts[action] += 1
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_data.append((state, action, reward))
                
                state = next_obs
                obs = next_obs
                first_action = False
            
            # Update statistics
            final_reward = episode_data[-1][2] if episode_data else 0
            self.total_return += final_reward
            self.episode_returns.append(final_reward)
            
            if final_reward > 0:
                self.wins += 1
            elif final_reward < 0:
                self.losses += 1
            else:
                self.ties += 1
            
            # Monte Carlo update (first-visit)
            G = 0
            visited = set()
            
            # Go through episode in reverse to calculate returns
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = self.gamma * G + reward
                
                # First-visit MC: only update if this is first time we see (state, action)
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns[(state, action)].append(G)
                    self.Q[(state, action)] = np.mean(self.returns[(state, action)])
            
            if (episode + 1) % 10000 == 0:
                avg_return = self.total_return / (episode + 1)
                win_rate = self.wins / (episode + 1) * 100
                print(f"Episode {episode + 1}/{self.episodes} | "
                      f"Avg Return: {avg_return:.4f} | "
                      f"Win Rate: {win_rate:.2f}%")
        
        print("=" * 70)
        print("Training complete!\n")
    
    def print_statistics(self):
        total_actions = sum(self.action_counts.values())
        total_games = self.wins + self.losses + self.ties
        
        print("\n" + "=" * 70)
        print("MONTE CARLO BLACKJACK - TRAINING STATISTICS")
        print("=" * 70)
        
        print("\nGAME OUTCOMES:")
        print(f"  Total Episodes: {total_games}")
        print(f"  Wins:           {self.wins} ({self.wins/total_games*100:.2f}%)")
        print(f"  Losses:         {self.losses} ({self.losses/total_games*100:.2f}%)")
        print(f"  Ties:           {self.ties} ({self.ties/total_games*100:.2f}%)")
        
        print("\nRETURNS:")
        print(f"  Average Return: {self.total_return/total_games:.4f}")
        print(f"  Std Dev:        {np.std(self.episode_returns):.4f}")
        
        print("\nACTION FREQUENCY:")
        print(f"  Total Actions Taken: {total_actions}")
        for action_id in range(5):
            count = self.action_counts.get(action_id, 0)
            frequency = count / total_actions * 100 if total_actions > 0 else 0
            name = self.action_names[action_id]
            print(f"  {name:12s}: {count:8d} ({frequency:5.2f}%)")
        
        print("\nLEARNED POLICY - KEY STATES:")
        
        # Check if we're using infinite deck env or 8 deck env for observations
        obs_dim = 3
        if self.Q:
            sample_state = next(iter(self.Q.keys()))[0]
            obs_dim = len(sample_state)
        
        # Define base states (player_sum, dealer_showing, usable_ace)
        base_states = [
            ((20, 10, 0), "Player 20 vs Dealer 10 (no ace)"),
            ((16, 10, 0), "Player 16 vs Dealer 10 (no ace)"),
            ((12, 6, 0), "Player 12 vs Dealer 6 (no ace)"),
            ((18, 1, 1), "Player 18 vs Dealer Ace (usable ace)"),
            ((11, 10, 0), "Player 11 vs Dealer 10 (no ace)"),
            ((10, 9, 0), "Player 10 vs Dealer 9 (no ace)"),
        ]
        
        for base_state, description in base_states:
            # For 8 deck env observations show average across all counts
            if obs_dim == 4:
                action_returns = defaultdict(list)
                for count in range(-2, 3):
                    state = base_state + (count,)
                    for action in range(5):
                        if (state, action) in self.Q:
                            action_returns[action].append(self.Q[(state, action)])
                
                if action_returns:
                    print(f"\n  {description}:")
                    q_vals = []
                    for action in range(5):
                        if action in action_returns:
                            avg_q = np.mean(action_returns[action])
                            action_name = self.action_names[action]
                            q_vals.append((action, action_name, avg_q))
                        else:
                            action_name = self.action_names[action]
                            q_vals.append((action, action_name, None))
                    
                    q_vals_with_values = [(a, n, v) for a, n, v in q_vals if v is not None]
                    if q_vals_with_values:
                        q_vals_with_values.sort(key=lambda x: x[2], reverse=True)
                        best_val = q_vals_with_values[0][2]
                        for action, action_name, q_val in q_vals:
                            if q_val is not None:
                                best = " *BEST*" if q_val == best_val else ""
                                print(f"    {action_name:12s}: {q_val:7.4f}{best}")
            else:
                # Infinite deck env observation handling
                state = base_state
                if any((state, a) in self.Q for a in range(5)):
                    print(f"\n  {description}:")
                    q_vals = []
                    for action in range(5):
                        if (state, action) in self.Q:
                            q_val = self.Q[(state, action)]
                            action_name = self.action_names[action]
                            q_vals.append((action, action_name, q_val))

                    q_vals.sort(key=lambda x: x[2], reverse=True)
                    for action, action_name, q_val in q_vals:
                        best = " *BEST*" if q_val == q_vals[0][2] else ""
                        print(f"    {action_name:12s}: {q_val:7.4f}{best}")
        
        print("\n" + "=" * 70)
    
    def evaluate(self, num_games=NUM_EVAL_GAMES):
        env = BlackjackEnv()
        
        wins = 0
        losses = 0
        ties = 0
        total_return = 0
        
        print(f"\nEvaluating trained policy over {num_games} games...")
        
        for game in range(num_games):
            obs, info = env.reset()
            state = obs
            terminated = False
            first_action = True
            
            while not terminated:
                valid_actions = self.get_valid_actions(env)
                # Use greedy policy (no exploration)
                q_values = [self.Q.get((state, a), 0) for a in valid_actions]
                action = valid_actions[np.argmax(q_values)]
                
                obs, reward, terminated, truncated, info = env.step(action)
                state = obs
                first_action = False
            
            total_return += reward
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                ties += 1
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (Greedy Policy)")
        print("=" * 70)
        print(f"  Games Played:   {num_games}")
        print(f"  Wins:           {wins} ({wins/num_games*100:.2f}%)")
        print(f"  Losses:         {losses} ({losses/num_games*100:.2f}%)")
        print(f"  Ties:           {ties} ({ties/num_games*100:.2f}%)")
        print(f"  Average Return: {total_return/num_games:.4f}")
        print("=" * 70)


def main():
    agent = MonteCarloBlackjack(episodes=NUM_EPISODES, epsilon=0.1, gamma=1.0)
    agent.train()
    
    agent.print_statistics()
    
    agent.evaluate(num_games=NUM_EVAL_GAMES)


if __name__ == "__main__":
    main()
