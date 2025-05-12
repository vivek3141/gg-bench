import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: move forward by 1, 2, or 3 positions (actions 0, 1, 2)
        self.action_space = spaces.Discrete(3)
        # Define observation space: positions of both players and the current player index (1 or 2)
        # Positions range from 0 to 10, current_player_index is 1 or 2
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]), high=np.array([10, 10, 2]), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = {1: 0, 2: 0}  # Starting positions of Player 1 and Player 2
        self.current_player = 1  # Player 1 starts the game
        self.done = False  # Game over flag
        observation = np.array(
            [self.positions[1], self.positions[2], self.current_player], dtype=np.int32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            observation = np.array(
                [self.positions[1], self.positions[2], self.current_player],
                dtype=np.int32,
            )
            return observation, 0, self.done, False, {}  # No reward, game already over

        # Validate action
        if action not in [0, 1, 2]:
            self.done = True
            observation = np.array(
                [self.positions[1], self.positions[2], self.current_player],
                dtype=np.int32,
            )
            return observation, -10, self.done, False, {}  # Invalid move

        # Move current player forward
        move_distance = action + 1  # Actions are 0,1,2 corresponding to moves of 1,2,3
        self.positions[self.current_player] += move_distance

        # Check for collision and bump opponent
        opponent = 2 if self.current_player == 1 else 1
        if self.positions[self.current_player] == self.positions[opponent]:
            # Bump opponent back to start
            self.positions[opponent] = 0

        # Check for victory
        if self.positions[self.current_player] >= 10:
            self.done = True
            reward = 1  # Current player wins
        else:
            reward = 0  # No reward if game continues

        # Prepare observation before switching player
        observation = np.array(
            [self.positions[1], self.positions[2], self.current_player], dtype=np.int32
        )

        # Switch to the next player if game is not over
        if not self.done:
            self.current_player = opponent

        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # Return observation, reward, done, truncated, info

    def render(self):
        track_display = []
        for position in range(11):  # Positions 0 to 10
            markers = []
            if self.positions[1] == position:
                markers.append("P1")
            if self.positions[2] == position:
                markers.append("P2")
            if markers:
                track_display.append("[" + ",".join(markers) + "]")
            else:
                track_display.append(str(position))
        track_visual = " ".join(track_display)
        current_player_info = f"Current turn: Player {self.current_player}"
        return f"{track_visual}\n{current_player_info}"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1, 2]  # Valid actions correspond to moving 1, 2, or 3 spaces
