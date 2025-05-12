import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting weights numbered 1 to 9 (indices 0-8)
        self.action_space = spaces.Discrete(9)

        # Observation space: array of 9 elements representing weights 1-9
        # Values: 0 (unclaimed), 1 (claimed by Player 1), -1 (claimed by Player 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the weights: 0 (unclaimed)
        self.weights = np.zeros(9, dtype=np.int8)
        # Player 1 starts (represented by 1), Player 2 is -1
        self.current_player = 1
        # Game over flag
        self.done = False
        return self.weights.copy(), {}  # Return observation and info

    def step(self, action):
        if self.weights[action] != 0 or self.done:
            # Invalid move: weight already claimed or game is over
            return self.weights.copy(), -10, True, False, {}

        # Get the weight value (weights are numbered 1 to 9)
        weight_value = action + 1
        player_total = self.get_player_total(self.current_player) + weight_value

        if player_total > 15:
            # Invalid move: cannot exceed 15 units
            return self.weights.copy(), -10, True, False, {}

        # Valid move: claim the weight
        self.weights[action] = self.current_player

        # Check for win condition
        if player_total == 15:
            # Current player wins
            self.done = True
            return self.weights.copy(), 1, True, False, {}

        # Switch to the next player
        next_player = -self.current_player

        # Check if the next player has any valid moves
        if not self.has_valid_moves(next_player):
            self.done = True
            # Determine winner based on total weights
            next_player_total = self.get_player_total(next_player)
            if player_total > next_player_total:
                # Current player wins
                reward = 1
            elif player_total < next_player_total:
                # Current player loses
                reward = -1
            else:
                # Draw
                reward = 0
            return self.weights.copy(), reward, True, False, {}

        # Continue the game
        self.current_player = next_player
        return self.weights.copy(), 0, False, False, {}

    def render(self):
        output = "Available Weights:\n"
        for i in range(9):
            if self.weights[i] == 0:
                output += f"{i+1} "
        output += "\n"
        output += f"Player 1 Total: {self.get_player_total(1)}\n"
        output += f"Player 2 Total: {self.get_player_total(-1)}\n"
        return output

    def valid_moves(self):
        """
        Returns a list of valid actions (indices) for the current player.
        """
        valid_actions = []
        player_total = self.get_player_total(self.current_player)
        for action in range(9):
            if self.weights[action] == 0:
                weight_value = action + 1
                if player_total + weight_value <= 15:
                    valid_actions.append(action)
        return valid_actions

    def get_player_total(self, player):
        """
        Calculates the total weight of the specified player.
        """
        indices = np.where(self.weights == player)[0]
        weights = indices + 1  # Convert indices to weight values
        return np.sum(weights)
