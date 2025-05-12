import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=20):
        super(CustomEnv, self).__init__()

        self.target_number = target_number

        # Define action and observation space
        # Two actions: 0 = Add 1, 1 = Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Observation space: [current_number, current_player]
        # current_number ranges from 1 to target_number
        # current_player can be 1 or 2
        low = np.array([1, 1], dtype=np.int32)
        high = np.array([self.target_number, 2], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.done = False
        self.current_player = 1  # Player 1 starts the game
        return np.array([self.current_number, self.current_player], dtype=np.int32), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if self.done:
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        if action == 0:
            new_number = self.current_number + 1
        elif action == 1:
            new_number = self.current_number * 2
        else:
            raise ValueError("Invalid action")

        if new_number == self.target_number:
            # Current player wins
            self.current_number = new_number
            reward = 1
            terminated = True
            self.done = True
        elif new_number > self.target_number:
            # Current player loses
            self.current_number = new_number
            reward = -10
            terminated = True
            self.done = True
        else:
            # Valid move, update current number and switch players
            self.current_number = new_number
            reward = 0
            terminated = False
            self.current_player = (
                3 - self.current_player
            )  # Switch between player 1 and 2

        observation = np.array(
            [self.current_number, self.current_player], dtype=np.int32
        )
        return observation, reward, terminated, False, {}

    def render(self):
        return f"Current Number: {self.current_number}, Target Number: {self.target_number}, Current Player: Player {self.current_player}"

    def valid_moves(self):
        valid_actions = []
        if self.current_number + 1 <= self.target_number:
            valid_actions.append(0)  # Add 1 is a valid move
        if self.current_number * 2 <= self.target_number:
            valid_actions.append(1)  # Multiply by 2 is a valid move
        return valid_actions
