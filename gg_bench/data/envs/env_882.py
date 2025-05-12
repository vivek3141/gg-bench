import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Move Marker A, 1 - Move Marker B
        self.action_space = spaces.Discrete(2)

        # Observation space: positions of Marker A and Marker B
        self.observation_space = spaces.Box(low=1, high=7, shape=(2,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.marker_A_pos = 1
        self.marker_B_pos = 7
        self.current_player = 1  # Player 1 starts
        self.done = False
        state = np.array([self.marker_A_pos, self.marker_B_pos], dtype=np.int8)
        return state, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            state = np.array([self.marker_A_pos, self.marker_B_pos], dtype=np.int8)
            return state, 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            state = np.array([self.marker_A_pos, self.marker_B_pos], dtype=np.int8)
            return state, -10, True, False, {}

        # Perform action
        reward = 0
        terminated = False
        truncated = False

        # Move the selected marker towards the other marker
        if action == 0:
            # Move Marker A
            if self.marker_A_pos < self.marker_B_pos:
                self.marker_A_pos += 1
        elif action == 1:
            # Move Marker B
            if self.marker_B_pos > self.marker_A_pos:
                self.marker_B_pos -= 1

        # Check for win
        if self.marker_A_pos == self.marker_B_pos:
            reward = 1  # Current player wins
            self.done = True
            terminated = True
        else:
            # Switch to the other player
            self.current_player = -self.current_player

        state = np.array([self.marker_A_pos, self.marker_B_pos], dtype=np.int8)
        return state, reward, terminated, truncated, {}

    def render(self):
        track = "Positions: "
        for pos in range(1, 8):
            track += f"{pos} "
        track += "\nMarkers:   "
        marker_line = ""
        for pos in range(1, 8):
            if self.marker_A_pos == self.marker_B_pos and self.marker_A_pos == pos:
                marker_line += "AB "
            elif self.marker_A_pos == pos:
                marker_line += "A  "
            elif self.marker_B_pos == pos:
                marker_line += "B  "
            else:
                marker_line += "   "
        return track + marker_line

    def valid_moves(self):
        moves = []
        # Check if Marker A can move
        if self.marker_A_pos < self.marker_B_pos:
            moves.append(0)  # Action 0: Move Marker A

        # Check if Marker B can move
        if self.marker_B_pos > self.marker_A_pos:
            moves.append(1)  # Action 1: Move Marker B

        return moves
