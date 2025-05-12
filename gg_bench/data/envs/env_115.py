import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are integers 0 to 4 mapping to numbers 1 to 5
        # So action_space has 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Observation space is the cumulative total represented as a single integer
        # We set the high value to a practical maximum (e.g., 100) for this game
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([100]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 0
        self.terminated = False
        self.truncated = False
        observation = np.array([self.cumulative_total], dtype=np.int32)
        info = {}
        return observation, info  # Return initial observation and empty info

    def step(self, action):
        # Map action to number between 1 and 5
        move_value = action + 1  # Actions 0-4 correspond to numbers 1-5

        # Check if action is valid
        if move_value not in range(1, 6):
            # Invalid action
            self.terminated = True
            reward = -10  # Penalty for invalid action
            observation = np.array([self.cumulative_total], dtype=np.int32)
            info = {}
            return observation, reward, self.terminated, self.truncated, info

        # Player's move
        self.cumulative_total += move_value

        # Check if the player loses
        if self.cumulative_total % 7 == 0:
            self.terminated = True
            reward = -10  # Penalty for losing
            observation = np.array([self.cumulative_total], dtype=np.int32)
            info = {}
            return observation, reward, self.terminated, self.truncated, info

        # Opponent's move (random strategy)
        opponent_move = self.opponent_move()
        self.cumulative_total += opponent_move

        # Check if the opponent loses
        if self.cumulative_total % 7 == 0:
            self.terminated = True
            reward = 1  # Reward for winning
            observation = np.array([self.cumulative_total], dtype=np.int32)
            info = {}
            return observation, reward, self.terminated, self.truncated, info

        # Game continues
        reward = -10  # Penalty for each move to encourage faster wins
        self.terminated = False
        observation = np.array([self.cumulative_total], dtype=np.int32)
        info = {}
        return observation, reward, self.terminated, self.truncated, info

    def opponent_move(self):
        # Opponent chooses a random number between 1 and 5
        opponent_action = np.random.randint(1, 6)
        return opponent_action

    def render(self):
        # Return a string representation of the game state
        return f"Cumulative total: {self.cumulative_total}"

    def valid_moves(self):
        # All actions from 0 to 4 are valid (representing numbers 1 to 5)
        return [action for action in range(5)]
