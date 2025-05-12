import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Add 1, 1 - Subtract 1, 2 - Multiply by 2
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Observation consists of [target_number, current_player_number, opponent_number]
        self.observation_space = spaces.Box(
            low=np.array([20, 1, 1], dtype=np.int32),
            high=np.array([30, 30, 30], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select target number between 20 and 30 inclusive
        self.target_number = self.np_random.integers(20, 31)

        # Both players start with current number 1
        self.player_numbers = [1, 1]  # Index 0 for Player 1, index 1 for Player 2

        # Start with Player 1
        self.current_player = 0  # 0 for Player 1, 1 for Player 2

        self.done = False
        self.truncated = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # Observation is [target_number, current_player_number, opponent_number]
        obs = np.array(
            [
                self.target_number,
                self.player_numbers[self.current_player],
                self.player_numbers[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        if self.done:
            # If game is over, return current observation
            return self._get_obs(), 0, self.done, self.truncated, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done, self.truncated, {}

        # Apply action to current player's number
        current_number = self.player_numbers[self.current_player]

        if action == 0:  # Add 1
            current_number += 1
        elif action == 1:  # Subtract 1
            current_number -= 1
        elif action == 2:  # Multiply by 2
            current_number *= 2

        # Update the player's number
        self.player_numbers[self.current_player] = current_number

        # Check for win/lose conditions
        if current_number == self.target_number:
            reward = 1  # Current player wins
            self.done = True
        elif current_number > self.target_number:
            reward = -10  # Current player loses
            self.done = True
        else:
            # Valid move, game continues
            reward = -10  # Negative reward per valid move
            self.current_player = 1 - self.current_player  # Switch player

        obs = self._get_obs()
        return obs, reward, self.done, self.truncated, {}

    def render(self):
        render_str = "--- Digit Duel ---\n"
        render_str += f"Target Number: {self.target_number}\n"
        render_str += f"Player 1's Number: {self.player_numbers[0]}\n"
        render_str += f"Player 2's Number: {self.player_numbers[1]}\n"
        render_str += f"Current Turn: Player {self.current_player + 1}\n"
        return render_str

    def valid_moves(self):
        moves = []
        current_number = self.player_numbers[self.current_player]
        # Add 1
        if (current_number + 1) <= self.target_number:
            moves.append(0)
        # Subtract 1
        if current_number > 1:
            moves.append(1)
        # Multiply by 2
        if (current_number * 2) <= self.target_number:
            moves.append(2)
        return moves
