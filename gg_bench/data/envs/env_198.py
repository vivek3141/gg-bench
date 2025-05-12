import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0: Add 1 (+1)
        # 1: Add 2 (+2)
        # 2: Multiply by 2 (×2)
        # 3: Subtract 1 (-1)

        self.action_space = spaces.Discrete(4)

        # Observation space: Current scores of both players
        # Shape: (2,), representing [Player 1's score, Player 2's score]
        self.observation_space = spaces.Box(low=0, high=20, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scores = np.array(
            [0, 0], dtype=np.int32
        )  # Scores for Player 0 and Player 1
        self.current_player = 0  # Start with Player 0
        self.done = False
        return self.scores.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # The game is over; any action is invalid
            return self.scores.copy(), -10, True, False, {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action; end the game
            self.done = True
            return self.scores.copy(), -10, True, False, {}

        # Apply the action
        player = self.current_player
        opponent = 1 - player
        score = self.scores[player]

        if action == 0:
            # Add 1
            new_score = score + 1
        elif action == 1:
            # Add 2
            new_score = score + 2
        elif action == 2:
            # Multiply by 2
            new_score = score * 2
        elif action == 3:
            # Subtract 1
            new_score = score - 1

        # Handle overruns
        if new_score > 20:
            new_score = 0

        # Ensure score doesn't go below zero
        new_score = max(0, new_score)

        # Update the score
        self.scores[player] = new_score

        # Check for win condition
        if new_score == 20:
            self.done = True
            return self.scores.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player = opponent

        # Reward for a valid move
        reward = -10

        return self.scores.copy(), reward, False, False, {}

    def render(self):
        output = f"Player 0's score: {self.scores[0]}\n"
        output += f"Player 1's score: {self.scores[1]}\n"
        if not self.done:
            output += f"Player {self.current_player}'s turn.\n"
        else:
            output += "Game over.\n"
            if self.scores[0] == 20:
                output += "Player 0 wins!\n"
            elif self.scores[1] == 20:
                output += "Player 1 wins!\n"
            else:
                output += "No winner.\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = [0, 1, 2]  # +1, +2, ×2 are always valid
        if self.scores[self.current_player] > 0:
            valid_actions.append(3)  # -1 is valid if current score > 0
        return valid_actions
