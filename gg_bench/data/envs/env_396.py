import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0-8: capture crystals 1-9, action 9: pass
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # - Indices 0-8: Available crystals (1 for available, 0 for captured)
        # - Index 9: Current player's points
        # - Index 10: Opponent's points
        # - Index 11: Current player indicator (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [0, 0, 0], dtype=np.int32),
            high=np.array([1] * 9 + [15, 15, 1], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_crystals = np.ones(
            9, dtype=np.int32
        )  # Crystals 1-9 are available
        self.player_points = [0, 0]  # Points for Player 0 and Player 1
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if action == 9:
            # Player passes their turn
            self.current_player = 1 - self.current_player
            return self._get_observation(), 0, False, False, {}

        # Capture the crystal
        crystal_value = action + 1  # Crystals are numbered from 1 to 9
        self.available_crystals[action] = 0
        self.player_points[self.current_player] += crystal_value

        # Check for win condition
        if self.player_points[self.current_player] == 15:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player
        return self._get_observation(), 0, False, False, {}

    def render(self):
        crystals = [str(i + 1) for i in range(9) if self.available_crystals[i] == 1]
        s = f"Available Crystals: {', '.join(crystals)}\n"
        s += f"Player 1 Points: {self.player_points[0]}\n"
        s += f"Player 2 Points: {self.player_points[1]}\n"
        s += f"Current Player: Player {self.current_player + 1}\n"
        return s

    def valid_moves(self):
        valid_moves = []
        current_points = self.player_points[self.current_player]
        for i in range(9):
            if self.available_crystals[i] == 1 and current_points + (i + 1) <= 15:
                valid_moves.append(i)
        if not valid_moves:
            # No valid crystals to capture, player must pass
            valid_moves.append(9)  # Action 9 represents 'pass'
        return valid_moves

    def _get_observation(self):
        observation = np.concatenate(
            [
                self.available_crystals,
                np.array(
                    [
                        self.player_points[self.current_player],
                        self.player_points[1 - self.current_player],
                        self.current_player,
                    ],
                    dtype=np.int32,
                ),
            ]
        )
        return observation
