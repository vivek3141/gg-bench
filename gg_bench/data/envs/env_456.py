import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible actions (numbers 1-9)
        self.action_space = spaces.Discrete(9)
        # Observation space consists of:
        # [Player1_HP, Player2_HP, Available_Numbers(9)]
        low_obs = np.array([-15, -15] + [0] * 9, dtype=np.int32)
        high_obs = np.array([15, 15] + [1] * 9, dtype=np.int32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_hp = 15
        self.player2_hp = 15
        # Available numbers: 1 means available, 0 means taken
        self.available_numbers = [1] * 9  # Indices 0-8 correspond to numbers 1-9
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no further moves are allowed
            return self._get_observation(), 0, True, False, {}

        # Convert action (0-8) to number (1-9)
        number = action + 1

        # Check if action is valid
        if action < 0 or action > 8 or self.available_numbers[action] == 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Apply damage to opponent
        if self.current_player == 1:
            self.player2_hp -= number
        else:
            self.player1_hp -= number

        # Remove number from available numbers
        self.available_numbers[action] = 0

        # Check for victory
        if self.current_player == 1 and self.player2_hp <= 0:
            self.done = True
            return self._get_observation(), 1, True, False, {}
        elif self.current_player == 2 and self.player1_hp <= 0:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if there are any available numbers left
        if sum(self.available_numbers) == 0:
            # Game cannot end in draw per rules, but no more moves can be made
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Continue game
        return self._get_observation(), -10, False, False, {}

    def render(self):
        # Generate a string representation of the game state
        state = "--- Number Battle ---\n"
        state += f"Player 1 HP: {self.player1_hp}\n"
        state += f"Player 2 HP: {self.player2_hp}\n"
        available_nums = [
            str(i + 1) for i in range(9) if self.available_numbers[i] == 1
        ]
        state += "Available Numbers: " + ",".join(available_nums) + "\n"
        state += f"Player {self.current_player}'s turn.\n"
        return state

    def valid_moves(self):
        # Return list of valid moves (action indices)
        return [i for i in range(9) if self.available_numbers[i] == 1]

    def _get_observation(self):
        # Observation includes player HPs and available numbers
        obs = np.array(
            [self.player1_hp, self.player2_hp] + self.available_numbers, dtype=np.int32
        )
        return obs
