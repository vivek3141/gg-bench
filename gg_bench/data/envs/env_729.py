import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0 to 4 correspond to bits 1 to 5 (left to right)
        self.action_space = spaces.Discrete(5)
        # Observation space: 5 bits for each player's number and current player indicator
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a random 5-bit secret target number
        self.target_number = np.random.randint(2, size=5)
        # Initialize both players' binary numbers to 00000
        self.player_numbers = np.zeros((2, 5), dtype=np.int8)
        # Randomly select the starting player (Player 0 or Player 1)
        self.current_player = np.random.randint(2)
        self.done = False
        # Return the initial observation and info
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            # If the game is over, ignore further actions
            return self._get_obs(), 0, True, False, {}

        # Validate the action
        if action < 0 or action > 4:
            # Invalid action results in immediate termination with a penalty
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Toggle the selected bit in the current player's binary number
        bit = action  # Bits are indexed from 0 to 4
        self.player_numbers[self.current_player, bit] ^= 1  # Toggle bit using XOR

        # Check for win condition
        if np.array_equal(self.player_numbers[self.current_player], self.target_number):
            self.done = True
            reward = 1  # Winning reward
            terminated = True
        else:
            reward = -10  # Penalty for a valid move that doesn't win
            terminated = False
            # Switch to the other player
            self.current_player = 1 - self.current_player

        # Return the observation, reward, and status flags
        observation = self._get_obs()
        return observation, reward, terminated, False, {}

    def render(self):
        # Generate a string representation of the game state
        player1_number = "".join(map(str, self.player_numbers[0]))
        player2_number = "".join(map(str, self.player_numbers[1]))
        target_number = "*****"  # The target number is hidden
        current_player = (
            self.current_player + 1
        )  # For display purposes (Player 1 or Player 2)

        render_str = f"Player 1 Number: {player1_number}\n"
        render_str += f"Player 2 Number: {player2_number}\n"
        render_str += f"Target Number: {target_number}\n"
        render_str += f"Current Player: Player {current_player}\n"
        return render_str

    def valid_moves(self):
        # All bit positions (0 to 4) are valid moves
        return [0, 1, 2, 3, 4]

    def _get_obs(self):
        # Construct the observation array
        observation = np.concatenate(
            (self.player_numbers.flatten(), [self.current_player])
        )
        return observation
