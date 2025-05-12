import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 for 'keep', 1 for 'give'
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation consists of:
        # [0]: Current player's score (25 to 50)
        # [1]: Opponent's score (25 to 50)
        # [2-10]: Remaining counts of tiles numbered 1 to 9 (values can be 0, 1, or 2)
        # [11]: Last drawn tile (0 if none, 1-9 if drawn)
        # [12]: Whose turn it is (0 or 1)
        low = np.array([25, 25] + [0] * 9 + [0, 0], dtype=np.int32)
        high = np.array([50, 50] + [2] * 9 + [9, 1], dtype=np.int32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the tile pool: two of each tile from 1 to 9 (total 18 tiles)
        self.tile_pool = [i for i in range(1, 10)] * 2  # Two of each tile
        np.random.shuffle(self.tile_pool)

        # Initialize counts of remaining tiles
        self.tile_counts = [2] * 9  # Indices 0-8 represent tiles 1-9

        # Initialize scores
        self.player_scores = [25, 25]  # Player 0 and Player 1

        # Set current player (Player 0 starts)
        self.current_player = 0

        # Last drawn tile (0 indicates no tile drawn yet)
        self.last_drawn_tile = 0

        # Game over flag
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # Build the observation array
        observation = np.zeros(13, dtype=np.int32)
        observation[0] = self.player_scores[
            self.current_player
        ]  # Current player's score
        observation[1] = self.player_scores[1 - self.current_player]  # Opponent's score
        observation[2:11] = self.tile_counts  # Remaining tile counts
        observation[11] = self.last_drawn_tile
        observation[12] = self.current_player  # Whose turn it is
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}  # Game is over

        # Check if the action is valid
        if action not in [0, 1]:
            # Invalid action
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Draw phase
        if len(self.tile_pool) == 0:
            # Reshuffle all tiles to form a new tile pool
            self.tile_pool = [i for i in range(1, 10)] * 2
            np.random.shuffle(self.tile_pool)
            self.tile_counts = [2] * 9

        # Draw the top tile
        self.last_drawn_tile = self.tile_pool.pop(0)
        # Update remaining tile counts
        self.tile_counts[self.last_drawn_tile - 1] -= 1

        # Decision phase
        current_player = self.current_player
        opponent_player = 1 - self.current_player

        reward = 0

        if action == 0:
            # Keep the tile
            recipient = current_player
            self.player_scores[recipient] += self.last_drawn_tile
        else:
            # Give the tile
            recipient = opponent_player
            self.player_scores[recipient] += self.last_drawn_tile

        # Bust check and win check for the recipient
        if self.player_scores[recipient] > 50:
            # Recipient busts
            self.player_scores[recipient] = 25
            if recipient == current_player:
                # Current player busts
                reward = -10  # Penalty for busting
        elif self.player_scores[recipient] == 50:
            # Recipient wins
            if recipient == current_player:
                # Current player wins
                reward = 1
            else:
                # Opponent wins
                reward = -1  # Penalty for causing opponent to win
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Prepare for the next turn
        # Reset the last drawn tile
        self.last_drawn_tile = 0
        # Switch turns
        self.current_player = opponent_player

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        # Return a string representation of the game state
        lines = []
        lines.append(f"Player {self.current_player}'s turn.")
        lines.append(
            f"Scores: Player 0: {self.player_scores[0]}, Player 1: {self.player_scores[1]}"
        )
        lines.append("Remaining tile counts:")
        for i in range(9):
            lines.append(f"Tile {i+1}: {self.tile_counts[i]}")
        lines.append(f"Last drawn tile: {self.last_drawn_tile}")
        return "\n".join(lines)

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1]
