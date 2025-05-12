import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = add one, 1 = double
        self.action_space = spaces.Discrete(2)
        # Observation: The current number (1 to 20)
        self.observation_space = spaces.Box(low=1, high=20, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Validate the action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10  # Penalty for invalid action
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:  # Add one
            new_number = self.current_number + 1
        elif action == 1:  # Double it
            new_number = self.current_number * 2

        # Check for win/lose conditions
        if new_number == 20:
            # Current player wins
            self.current_number = new_number
            self.done = True
            reward = 1  # Reward for winning
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        elif new_number > 20:
            # Current player loses
            self.current_number = new_number
            self.done = True
            reward = -10  # Penalty for losing
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Valid move, continue game
            self.current_number = new_number
            self.current_player *= -1  # Switch player
            reward = 0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return (
            f"Current Number: {self.current_number}, "
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )

    def valid_moves(self):
        valid_actions = []
        if self.current_number + 1 <= 20:
            valid_actions.append(0)  # Add one
        if self.current_number * 2 <= 20:
            valid_actions.append(1)  # Double it
        return valid_actions
