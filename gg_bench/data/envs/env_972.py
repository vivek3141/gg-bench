import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 20 possible actions (numbers from 1 to 20)
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # - An array of length 21
        # - Indices 0-19 represent the availability of numbers 1-20 (1 if available, 0 if taken)
        # - Index 20 represents the current number (0 if None, else 1-20)
        self.observation_space = spaces.Box(low=0, high=20, shape=(21,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 1 to 20 are initially available (1 means available)
        self.number_pool = np.ones(20, dtype=np.int32)
        # No current number at the start of the game
        self.current_number = 0  # 0 indicates None
        # Player 1 starts the game
        self.current_player = 1
        # Game is not over
        self.done = False
        # Prepare the observation
        observation = self._get_obs()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        number = action + 1  # Convert action index to actual number (1-20)

        # Check if the selected number is available
        if self.number_pool[action] == 0:
            # Invalid move: number already taken
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Reward, terminated, truncated, info

        # Determine if the move is valid based on the game rules
        if self.current_number == 0:
            # First turn: any number can be selected
            valid = True
        else:
            # Subsequent turns: the number must be a factor or multiple of the current number
            if self.current_number % number == 0 or number % self.current_number == 0:
                valid = True
            else:
                valid = False

        if not valid:
            # Invalid move: doesn't satisfy divisibility rules
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move: update the game state
        self.number_pool[action] = 0  # Remove the number from the pool
        self.current_number = number  # Update the current number

        # Check if the next player has any valid moves
        valid_moves_next_player = self.valid_moves()
        if not valid_moves_next_player:
            # Next player cannot make a valid move; current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Switch to the other player and continue the game
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        return (
            self._get_obs(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Generate a visual representation of the game state
        state_str = f"Current Player: Player {self.current_player}\n"
        state_str += f"Current Number: {self.current_number if self.current_number != 0 else 'None'}\n"
        available_numbers = [str(i + 1) for i in range(20) if self.number_pool[i] == 1]
        state_str += f"Available Numbers: {', '.join(available_numbers)}\n"
        return state_str

    def valid_moves(self, current_number=None):
        # Return a list of valid move indices based on the current number
        if current_number is None:
            current_number = self.current_number
        valid_actions = []
        for i in range(20):
            if self.number_pool[i] == 1:
                number = i + 1  # Convert index to number
                if current_number == 0:
                    # First turn: any number is valid
                    valid_actions.append(i)
                else:
                    # Check if the number is a factor or multiple of the current number
                    if current_number % number == 0 or number % current_number == 0:
                        valid_actions.append(i)
        return valid_actions

    def _get_obs(self):
        # Create the observation array
        observation = np.append(self.number_pool, self.current_number)
        return observation
