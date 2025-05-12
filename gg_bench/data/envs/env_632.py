import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions correspond to moving 1, 2, or 4 positions
        self.action_space = spaces.Discrete(3)

        # Observation includes positions of both players (0-20), skip status of both players (0 or 1), current player (1 or 2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 1], dtype=np.int32),
            high=np.array([20, 20, 1, 1, 2], dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = [0, 0]  # Positions of Player 1 and Player 2
        self.player_skip = [False, False]  # Skip status of each player
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        # Handle skipped turn
        if self.player_skip[self.current_player]:
            self.player_skip[self.current_player] = False  # Reset skip status
            self._next_player()
            return self._get_obs(), 0, False, False, {}  # Skip turn without action

        # Check if action is valid
        if action not in self.valid_moves():
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move ends the game

        # Map action to movement
        move_options = [1, 2, 4]
        move = move_options[action]

        # Update player's position
        current_pos = self.player_positions[self.current_player]
        new_pos = current_pos + move

        # Check for invalid move (overstepping position 20)
        if new_pos > 20:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move ends the game

        self.player_positions[self.current_player] = new_pos

        # Check for win condition
        if new_pos == 20:
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Check for special positions
        if new_pos in [5, 10, 15]:
            self.player_skip[self.current_player] = True  # Player must skip next turn

        # Switch to next player's turn
        self._next_player()

        return self._get_obs(), 0, False, False, {}  # Continue game

    def render(self):
        # Return a visual representation of the environment state as a string
        player1_status = "(will skip next turn)" if self.player_skip[0] else ""
        player2_status = "(will skip next turn)" if self.player_skip[1] else ""
        return (
            f"Player 1 is at position {self.player_positions[0]} {player1_status}\n"
            f"Player 2 is at position {self.player_positions[1]} {player2_status}\n"
            f"It's Player {self.current_player + 1}'s turn.\n"
        )

    def valid_moves(self):
        if self.done:
            return []

        if self.player_skip[self.current_player]:
            return []  # No valid moves if player must skip turn

        move_options = [1, 2, 4]
        valid_actions = []
        current_pos = self.player_positions[self.current_player]
        for idx, move in enumerate(move_options):
            if current_pos + move <= 20:
                valid_actions.append(idx)
        return valid_actions

    def _get_obs(self):
        return np.array(
            [
                self.player_positions[0],
                self.player_positions[1],
                int(self.player_skip[0]),
                int(self.player_skip[1]),
                self.current_player + 1,
            ],
            dtype=np.int32,
        )

    def _next_player(self):
        # Switch to the other player
        self.current_player = 1 - self.current_player
