import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space includes number counts (1-9) and player scores (Player 1 and Player 2)
        # Total of 11 elements: 9 counts + 2 scores
        self.observation_space = spaces.Box(
            low=0, high=150, shape=(11,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize number counts for numbers 1 to 9
        self.number_counts = np.zeros(9, dtype=np.int32)
        # Initialize player scores: [Player 1's score, Player 2's score]
        self.player_scores = np.zeros(2, dtype=np.int32)
        # Set current player: 0 for Player 1, 1 for Player 2
        self.current_player = 0
        # Game over flag
        self.done = False
        # Get initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            # If game is over, no further moves are allowed
            return self._get_observation(), 0, True, False, {}
        # Convert action (0-8) to number (1-9)
        number_chosen = action + 1

        # Check if the move is valid
        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move: current player loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Increment the count for the chosen number
        self.number_counts[action] += 1
        total_selections = self.number_counts[action]

        # Calculate points gained
        points_gained = number_chosen * total_selections

        # Update current player's score
        self.player_scores[self.current_player] += points_gained

        # Check for win condition
        if self.player_scores[self.current_player] == 100:
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check for loss condition (score exceeds 100)
        if self.player_scores[self.current_player] > 100:
            # Current player loses
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            # Next player has no valid moves: current player wins
            reward = 1
            self.done = True
            # Switch back to the winning player for accurate observation
            self.current_player = 1 - self.current_player
            return self._get_observation(), reward, True, False, {}

        # Game continues
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the game state
        output = "Current Scores: Player 1 - {}, Player 2 - {}\n".format(
            self.player_scores[0], self.player_scores[1]
        )
        output += "Number Counts: "
        output += ", ".join(
            ["{}({})".format(i + 1, self.number_counts[i]) for i in range(9)]
        )
        output += "\n"
        output += "Player {}'s turn.\n".format(self.current_player + 1)
        return output

    def valid_moves(self):
        # Return a list of valid actions (numbers that won't cause the player to exceed 100)
        valid_moves = []
        for action in range(9):
            number_chosen = action + 1  # Convert action to number (1-9)
            future_count = self.number_counts[action] + 1  # Include current selection
            points_gained = number_chosen * future_count
            future_score = self.player_scores[self.current_player] + points_gained
            if future_score <= 100:
                valid_moves.append(action)
        return valid_moves

    def _get_observation(self):
        # Concatenate number counts and player scores into a single observation array
        obs = np.concatenate((self.number_counts, self.player_scores))
        return obs
