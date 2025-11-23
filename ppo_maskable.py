import numpy as np
import torch as th
#from blackjack_env import BlackjackEnv
from blackjack_8deck_env import Blackjack8DeckEnv as BlackjackEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict

# Unlike monte_carlo.py and qlearning.py, ppo_maskable.py is a difficult algorithm to implement so we used the SB3 library

NUM_TIMESTEPS = 10000
NUM_EVAL_GAMES = 10000

class FlattenObservationWrapper(gym.ObservationWrapper):
    # Wrapper to convert Tuple observation space to Box observation space for SB3
    def __init__(self, env):
        super().__init__(env)
        
        # Auto-detect observation space dimensions (3 for original, 4 for 8-deck with count)
        obs_shape = env.observation_space.spaces
        obs_dim = len(obs_shape)
        
        if obs_dim == 3:
            # Original environment: (player_sum, dealer_showing, usable_ace)
            self.observation_space = spaces.Box(
                low=np.array([4, 1, 0], dtype=np.float32),
                high=np.array([22, 10, 1], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # 8-deck environment: (player_sum, dealer_showing, usable_ace, true_count)
            self.observation_space = spaces.Box(
                low=np.array([4, 1, 0, -2], dtype=np.float32),
                high=np.array([22, 10, 1, 2], dtype=np.float32),
                dtype=np.float32
            )
    
    def observation(self, obs):
        # Convert tuple observation to numpy array
        return np.array(obs, dtype=np.float32)


def mask_fn(env: gym.Env) -> np.ndarray:
    # Function to get action mask from the environment
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    return base_env.action_masks()


class ActionTrackingCallback(BaseCallback):
    # Callback to track action distribution during training
    def __init__(self, verbose=0):
        super().__init__(verbose)
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
        
    def _on_step(self) -> bool:
        # Track actions taken
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]
            self.action_counts[action] += 1
        return True
    
    def get_action_stats(self):
        return dict(self.action_counts), self.action_names


class TrainingStatsCallback(BaseCallback):
    # Callback to track training statistics (wins, losses, ties, returns)
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.total_return = 0.0
        self.episodes_completed = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            reward = self.locals.get('rewards', [0])[0]
            self.episode_returns.append(reward)
            self.total_return += reward
            self.episodes_completed += 1
            
            if reward > 0:
                self.wins += 1
            elif reward < 0:
                self.losses += 1
            else:
                self.ties += 1
                
        return True
    
    def get_stats(self):
        return {
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'total_return': self.total_return,
            'episode_returns': self.episode_returns,
            'episodes_completed': self.episodes_completed
        }


def make_env():
    env = BlackjackEnv()
    env = FlattenObservationWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def train_maskable_ppo(total_timesteps=NUM_TIMESTEPS):
    print(f"Training Maskable PPO agent for {total_timesteps} timesteps...")
    print("=" * 70)
    
    env = DummyVecEnv([make_env])
    
    action_callback = ActionTrackingCallback()
    stats_callback = TrainingStatsCallback()
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=None
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[action_callback, stats_callback],
        progress_bar=True
    )
    
    print("\n" + "=" * 70)
    print("Training complete!\n")
    
    model.save("models/ppo_blackjack/maskable_ppo_blackjack")
    print("Model saved to: models/ppo_blackjack/maskable_ppo_blackjack\n")
    
    return model, action_callback, stats_callback


def print_training_statistics(action_callback, stats_callback):
    action_counts, action_names = action_callback.get_action_stats()
    stats = stats_callback.get_stats()
    
    total_actions = sum(action_counts.values())
    total_games = stats['episodes_completed']
    
    print("\n" + "=" * 70)
    print("MASKABLE PPO BLACKJACK - TRAINING STATISTICS")
    print("=" * 70)
    
    print("\nGAME OUTCOMES:")
    print(f"  Total Episodes: {total_games}")
    if total_games > 0:
        print(f"  Wins:           {stats['wins']} ({stats['wins']/total_games*100:.2f}%)")
        print(f"  Losses:         {stats['losses']} ({stats['losses']/total_games*100:.2f}%)")
        print(f"  Ties:           {stats['ties']} ({stats['ties']/total_games*100:.2f}%)")
    
    print("\nRETURNS:")
    if total_games > 0:
        print(f"  Average Return: {stats['total_return']/total_games:.4f}")
        print(f"  Std Dev:        {np.std(stats['episode_returns']):.4f}")
    
    print("\nACTION FREQUENCY:")
    print(f"  Total Actions Taken: {total_actions}")
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        frequency = count / total_actions * 100 if total_actions > 0 else 0
        name = action_names[action_id]
        print(f"  {name:12s}: {count:8d} ({frequency:5.2f}%)")
    
    print("\n" + "=" * 70)


def evaluate_model(model, num_games=NUM_EVAL_GAMES):
    env = make_env()
    
    wins = 0
    losses = 0
    ties = 0
    total_return = 0.0
    
    action_counts = {
        0: 0,  # stand
        1: 0,  # hit
        2: 0,  # double
        3: 0,  # split
        4: 0   # insurance
    }
    action_names = {
        0: "Stand",
        1: "Hit",
        2: "Double",
        3: "Split",
        4: "Insurance"
    }

    state_action_returns = defaultdict(list)
    
    print(f"\nEvaluating trained policy over {num_games} games...")
    
    for game in range(num_games):
        obs, info = env.reset()
        terminated = False
        episode_reward = 0
        episode_data = []
        
        while not terminated:
            # Get action mask from the environment
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            action_mask = base_env.action_masks()
            
            state = tuple(obs.astype(int))
            
            # Predict action using the trained model
            action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)
            action = int(action)
            
            action_counts[action] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_data.append((state, action, reward))
            episode_reward = reward
        
        total_return += episode_reward
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            ties += 1

        for state, action, _ in episode_data:
            state_action_returns[(state, action)].append(episode_reward)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Deterministic Policy)")
    print("=" * 70)
    print(f"  Games Played:   {num_games}")
    print(f"  Wins:           {wins} ({wins/num_games*100:.2f}%)")
    print(f"  Losses:         {losses} ({losses/num_games*100:.2f}%)")
    print(f"  Ties:           {ties} ({ties/num_games*100:.2f}%)")
    print(f"  Average Return: {total_return/num_games:.4f}")
    print("=" * 70)
    
    total_actions = sum(action_counts.values())
    print("\nEVALUATION ACTION FREQUENCY:")
    print(f"  Total Actions Taken: {total_actions}")

    for action_id in range(5):
        count = action_counts.get(action_id, 0)
        frequency = count / total_actions * 100 if total_actions > 0 else 0
        name = action_names[action_id]
        print(f"  {name:12s}: {count:8d} ({frequency:5.2f}%)")
    
    print("\nLEARNED POLICY - KEY STATES:")
    
    # Check if we're using infinite deck env or 8 deck env for observations
    obs_dim = 3
    if state_action_returns:
        sample_state = next(iter(state_action_returns.keys()))[0]
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
        if obs_dim == 4:
            # 8 deck env observation handling
            action_returns = defaultdict(list)
            for count in range(-2, 3):
                state = base_state + (count,)
                for action in range(5):
                    if (state, action) in state_action_returns:
                        action_returns[action].extend(state_action_returns[(state, action)])
            
            if action_returns:
                print(f"\n  {description}:")
                q_vals = []
                for action in range(5):
                    if action in action_returns:
                        avg_return = np.mean(action_returns[action])
                        action_name = action_names[action]
                        q_vals.append((action, action_name, avg_return))
                    else:
                        action_name = action_names[action]
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
            # infinite deck env observation handling
            state = base_state
            if any((state, a) in state_action_returns for a in range(5)):
                print(f"\n  {description}:")
                q_vals = []
                for action in range(5):
                    if (state, action) in state_action_returns:
                        avg_return = np.mean(state_action_returns[(state, action)])
                        action_name = action_names[action]
                        q_vals.append((action, action_name, avg_return))
                
                q_vals.sort(key=lambda x: x[2], reverse=True)
                for action, action_name, q_val in q_vals:
                    best = " *BEST*" if q_val == q_vals[0][2] else ""
                    print(f"    {action_name:12s}: {q_val:7.4f}{best}")
    
    print("\n" + "=" * 70)
    
    env.close()


def main():
    model, action_callback, stats_callback = train_maskable_ppo(total_timesteps=NUM_TIMESTEPS)
    
    print_training_statistics(action_callback, stats_callback)
    
    evaluate_model(model, num_games=NUM_EVAL_GAMES)


if __name__ == "__main__":
    main()
