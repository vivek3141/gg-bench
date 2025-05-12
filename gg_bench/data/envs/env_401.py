import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to divisors from 2 to 99 (indices 0 to 97)
        self.action_space = spaces.Discrete(98)
        # Observation: [current_number, current_player]
        # current_number ranges from 1 to 100
        # current_player is -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([100, 1]), dtype=np.int32
        )

        self.current_number = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 100
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array(
            [self.current_number, self.current_player], dtype=np.int32
        )
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        if self.done:
            raise RuntimeError("Game is already over. Please reset the environment.")

        proposed_divisor = action + 2  # Actions correspond to divisors from 2 upwards

        # Check if current player has any valid moves
        valid_divisors = [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]
        if not valid_divisors:
            # Current player cannot make a move and loses
            reward = -1
            self.done = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Check if the action is valid
        if proposed_divisor not in valid_divisors:
            # Invalid action
            reward = -10
            self.done = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Valid action, apply it
        self.current_number = self.current_number // proposed_divisor

        # Check if current number is 1
        if self.current_number == 1:
            # Current player wins
            reward = 1
            self.done = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Check if opponent can make a move
        opponent_valid_divisors = [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]
        if not opponent_valid_divisors:
            # Opponent cannot make a move, current player wins
            reward = 1
            self.done = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Switch current player
        self.current_player *= -1
        reward = 0
        self.done = False
        observation = np.array(
            [self.current_number, self.current_player], dtype=np.int32
        )
        info = {}
        return observation, reward, self.done, False, info

    def render(self):
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return (
            f"Current number: {self.current_number}\n" f"Current turn: {player_str}\n"
        )

    def valid_moves(self):
        valid_divisors = [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]
        valid_actions = [
            d - 2 for d in valid_divisors
        ]  # Convert divisors to action indices
        return valid_actions
