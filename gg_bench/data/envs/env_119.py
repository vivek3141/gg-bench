import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers from 2 to 20 inclusive (indices 0 to 18)
        self.action_space = spaces.Discrete(19)

        # Observation space: array of 20 elements
        # Elements 0 to 18: pool numbers from 2 to 20 inclusive, 1 if in pool, 0 if removed
        # Element 19: last number chosen, scaled between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the pool: numbers from 2 to 20 inclusive
        self.pool = np.ones(
            19, dtype=np.float32
        )  # All numbers are initially in the pool
        self.last_number = 0  # No last number at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Observation
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action index to actual number
        number = action + 2  # Action 0 corresponds to number 2

        # Check if number is in pool
        if self.pool[action] == 0:
            # Invalid move: number not in pool
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, True, False, {}

        # Check if the selected number is a valid move
        # On the first move, any number is valid
        if self.last_number != 0:
            if number % self.last_number != 0 and self.last_number % number != 0:
                # Invalid move
                self.done = True
                reward = -10  # Penalty for invalid move
                return self._get_obs(), reward, True, False, {}

        # Valid move
        self.pool[action] = 0  # Remove the number from the pool
        self.last_number = number

        # Check if the next player has any valid moves
        next_player_valid_moves = self.get_valid_moves()
        if not next_player_valid_moves:
            # Next player cannot make a valid move, current player wins
            self.done = True
            reward = 1  # Reward for winning the game
            return self._get_obs(), reward, True, False, {}
        else:
            # Swap player
            self.current_player *= -1
            # Game continues
            reward = 0
            return self._get_obs(), reward, False, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        pool_numbers = [str(i + 2) for i in range(19) if self.pool[i]]
        pool_str = ", ".join(pool_numbers)
        state_str = f"Current player: Player {1 if self.current_player == 1 else 2}\n"
        state_str += f"Last number chosen: {self.last_number if self.last_number != 0 else 'None'}\n"
        state_str += f"Numbers remaining in pool: {pool_str}"
        return state_str

    def valid_moves(self):
        # Return a list of valid moves as indices of the action_space
        return self.get_valid_moves()

    def get_valid_moves(self):
        # Returns a list of valid action indices for the next player
        valid_moves = []
        # If game is over, no valid moves
        if self.done:
            return valid_moves
        last_number = self.last_number
        for i in range(19):
            if self.pool[i]:
                number = i + 2
                if number % last_number == 0 or last_number % number == 0:
                    valid_moves.append(i)
        return valid_moves

    def _get_obs(self):
        # Return the current observation
        # Scale last_number between 0 and 1
        if self.last_number == 0:
            scaled_last_number = 0.0
        else:
            scaled_last_number = (self.last_number - 2) / 18.0  # Scale between 0 and 1
        obs = np.concatenate(
            [self.pool, np.array([scaled_last_number], dtype=np.float32)]
        )
        return obs
