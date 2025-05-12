import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # There are 15 possible actions, corresponding to numbers 1 to 15
        self.action_space = spaces.Discrete(15)

        # Observation space includes availability of numbers 1-15 and parity requirement
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 1 to 15 are initially available
        self.available_numbers = np.ones(15, dtype=np.int8)

        # Parity requirement: -1 indicates no parity required (first move)
        self.parity_required = -1

        # Current player: 1 or -1; environment manages the turn internally
        self.current_player = 1

        # Game completion flag
        self.done = False

        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Combine available numbers and parity requirement into a single observation array
        obs = np.concatenate(
            (self.available_numbers, [self.parity_required]), axis=None
        )
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game already completed

        terminated = False
        truncated = False
        reward = 0
        number = action + 1  # Map action index to actual number

        # Check for invalid actions
        if action < 0 or action >= 15 or self.available_numbers[action] != 1:
            # Invalid action: number not available or out of bounds
            terminated = True
            reward = -10
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}

        if self.parity_required != -1:
            # Enforce parity rule from the second turn onwards
            required_parity = self.parity_required
            if number % 2 != required_parity:
                # Invalid action: parity rule not respected
                terminated = True
                reward = -10
                self.done = True
                return self._get_obs(), reward, terminated, truncated, {}

        # Valid move execution
        self.available_numbers[action] = 0  # Remove the selected number

        # Update parity requirement for the next turn
        if number % 2 == 0:
            self.parity_required = 1  # Next player must choose an odd number
        else:
            self.parity_required = 0  # Next player must choose an even number

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move; current player wins
            terminated = True
            reward = 1
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}
        else:
            # Switch turn to the next player
            self.current_player *= -1
            return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # Visual representation of the game state
        numbers_state = []
        for i in range(15):
            if self.available_numbers[i] == 1:
                numbers_state.append(str(i + 1))
            else:
                numbers_state.append("X")  # Number has been selected
        if self.parity_required == -1:
            parity_str = "No parity constraint"
        elif self.parity_required == 0:
            parity_str = "Next number must be even"
        else:
            parity_str = "Next number must be odd"
        state_str = "Available numbers: " + ", ".join(numbers_state)
        state_str += "\nParity requirement: " + parity_str
        print(state_str)

    def valid_moves(self):
        # Generate a list of valid action indices based on the parity rule and available numbers
        valid_actions = []
        for action in range(15):
            if self.available_numbers[action] == 1:
                number = action + 1
                if self.parity_required == -1:
                    # First move, no parity requirement
                    valid_actions.append(action)
                else:
                    required_parity = self.parity_required
                    if number % 2 == required_parity:
                        valid_actions.append(action)
        return valid_actions
