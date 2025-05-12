import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Double own number, 1 - Add opponent's number
        self.action_space = spaces.Discrete(2)

        # Observation space: [current player's number, opponent's number]
        # Both numbers range from 1 to 100
        self.observation_space = spaces.Box(low=1, high=100, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting numbers
        self.player1_number = 1
        self.player2_number = 1
        # Player 1 starts
        self.current_player = 1  # 1 for Player 1, 2 for Player 2
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action, penalize and end the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Get current player's and opponent's numbers
        if self.current_player == 1:
            own_number = self.player1_number
            opp_number = self.player2_number
        else:
            own_number = self.player2_number
            opp_number = self.player1_number

        # Apply action
        if action == 0:
            # Double own number
            new_number = own_number + own_number
        else:
            # Add opponent's number
            new_number = own_number + opp_number

        # Check if new number exceeds 100 (should not happen due to valid_moves check)
        if new_number > 100:
            # Invalid action, penalize and end the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Update current player's number
        if self.current_player == 1:
            self.player1_number = new_number
        else:
            self.player2_number = new_number

        # Check for victory
        if new_number == 100:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if next player has valid moves
        if len(self.valid_moves()) == 0:
            # Next player cannot move, current player wins
            self.done = True
            # Switch back to previous player for accurate observation
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Valid move made, penalize to encourage quick victory
        reward = -10
        return self._get_observation(), reward, False, False, {}

    def render(self):
        return (
            f"Current State:\n"
            f"Player 1's Number: {self.player1_number}\n"
            f"Player 2's Number: {self.player2_number}\n"
            f"Player {self.current_player}'s Turn"
        )

    def valid_moves(self):
        if self.done:
            return []

        # Get current player's and opponent's numbers
        if self.current_player == 1:
            own_number = self.player1_number
            opp_number = self.player2_number
        else:
            own_number = self.player2_number
            opp_number = self.player1_number

        valid_actions = []
        # Check if doubling own number is valid
        if own_number + own_number <= 100:
            valid_actions.append(0)
        # Check if adding opponent's number is valid
        if own_number + opp_number <= 100:
            valid_actions.append(1)
        return valid_actions

    def _get_observation(self):
        # Observation from the current player's perspective
        if self.current_player == 1:
            return np.array([self.player1_number, self.player2_number], dtype=np.int32)
        else:
            return np.array([self.player2_number, self.player1_number], dtype=np.int32)
