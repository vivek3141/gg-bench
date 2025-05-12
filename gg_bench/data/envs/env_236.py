import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions 0-7 correspond to multipliers 2-9
        self.action_space = spaces.Discrete(8)

        # Define observation space: current total (1-100) and current player (1 or 2)
        self.observation_space = spaces.Box(
            low=np.array([1, 1]), high=np.array([100, 2]), dtype=np.int32
        )

        self.current_total = None
        self.current_player = None
        self.done = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array(
            [self.current_total, self.current_player], dtype=np.int32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            observation = np.array(
                [self.current_total, self.current_player], dtype=np.int32
            )
            return observation, 0, True, False, {}  # No reward, terminated

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            reward = -10
            self.done = True
            observation = np.array(
                [self.current_total, self.current_player], dtype=np.int32
            )
            return (
                observation,
                reward,
                True,
                False,
                {},
            )  # Game over due to invalid action

        # Map action to multiplier (action 0 corresponds to 2)
        multiplier = action + 2

        # Calculate new total
        new_total = self.current_total * multiplier

        if new_total == 100:
            # Current player wins
            self.current_total = new_total
            reward = 1
            self.done = True
            observation = np.array(
                [self.current_total, self.current_player], dtype=np.int32
            )
            return observation, reward, True, False, {}
        elif new_total > 100:
            # Current player loses (should not happen with valid actions)
            reward = -10
            self.current_total = new_total
            self.done = True
            observation = np.array(
                [self.current_total, self.current_player], dtype=np.int32
            )
            return observation, reward, True, False, {}
        else:
            # Valid move, game continues
            reward = -10  # Reward for a valid move
            self.current_total = new_total
            # Switch player
            self.current_player = 1 if self.current_player == 2 else 2
            observation = np.array(
                [self.current_total, self.current_player], dtype=np.int32
            )
            return observation, reward, False, False, {}

    def render(self):
        return (
            f"Current Total: {self.current_total}, Player {self.current_player}'s turn."
        )

    def valid_moves(self):
        # Return list of valid action indices (0-7 corresponding to multipliers 2-9)
        valid_actions = []
        for action in range(8):
            multiplier = action + 2
            new_total = self.current_total * multiplier
            if new_total <= 100:
                valid_actions.append(action)
        return valid_actions
