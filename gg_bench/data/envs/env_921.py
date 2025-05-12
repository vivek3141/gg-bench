import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum number for the game and the starting number
        self.max_number = 100  # Maximum possible current number
        self.starting_number = 30  # Default starting number

        # Action space: actions correspond to divisors from 2 up to max_number
        # Actions are indices from 0 to max_number - 2, where action n corresponds to divisor n + 2
        self.action_space = spaces.Discrete(
            self.max_number - 1
        )  # Excludes 1 as a divisor

        # Observation space: [current_number, current_player]
        # current_number ranges from 1 to max_number
        # current_player is 1 or 2
        self.observation_space = spaces.Box(
            low=np.array([1, 1]),
            high=np.array([self.max_number, 2]),
            shape=(2,),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and empty info dict

    def _get_observation(self):
        """Helper method to construct the current observation."""
        return np.array([self.current_number, self.current_player], dtype=np.int32)

    def __proper_divisors(self, n):
        """Returns a list of proper divisors of n, excluding 1 and n itself."""
        return [d for d in range(2, n) if n % d == 0]

    def valid_moves(self):
        """
        Returns a list of valid actions (indices) for the current state.
        Actions correspond to proper divisors of the current number.
        """
        divisors = self.__proper_divisors(self.current_number)
        valid_actions = [d - 2 for d in divisors]  # Map divisors to action indices
        return valid_actions

    def step(self, action):
        if self.done:
            # If the game is already over
            return self._get_observation(), 0, True, False, {}

        # Map action index to actual divisor
        divisor = action + 2  # action 0 corresponds to divisor 2

        # Validate the action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_observation(), reward, True, False, {}
        else:
            # Apply the valid action
            self.current_number = self.current_number // divisor

            # Check if the opponent can make a move
            opponent_divisors = self.__proper_divisors(self.current_number)
            if len(opponent_divisors) == 0:
                # Opponent cannot move; current player wins
                self.done = True
                reward = 1  # Reward for winning
                return self._get_observation(), reward, True, False, {}
            else:
                # Switch to opponent's turn
                self.current_player = 1 if self.current_player == 2 else 2
                reward = 0  # No immediate reward
                return self._get_observation(), reward, False, False, {}

    def render(self):
        """Returns a string representation of the current game state."""
        s = f"Player {self.current_player}'s Turn:\n"
        s += f"Current Number: {self.current_number}\n"
        divisors = self.__proper_divisors(self.current_number)
        if divisors:
            s += f"Proper Divisors (excluding 1 and {self.current_number}): {', '.join(map(str, divisors))}\n"
        else:
            s += f"No proper divisors available. {self.current_number} is a prime number or 1.\n"
        return s
