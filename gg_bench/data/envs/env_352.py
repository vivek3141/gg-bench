import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions:
        # 0 - Build 1 unit
        # 1 - Build 2 units
        # 2 - Build 4 units
        # 3 - Attack (remove 3 units from opponent's tower)
        self.action_space = spaces.Discrete(4)

        # Define observation space: [current player's tower height, opponent's tower height]
        self.observation_space = spaces.Box(low=0, high=15, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.towers = [0, 0]  # [Player 1 tower height, Player 2 tower height]
        self.current_player = 0  # Player 1 starts (index 0)
        self.done = False  # Game over flag
        return np.array(self._get_obs()), {}  # Observation and info dictionary

    def step(self, action):
        if self.done:
            return np.array(self._get_obs()), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            # Invalid move
            return (
                np.array(self._get_obs()),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        curr = self.current_player
        opp = 1 - curr
        reward = 0
        terminated = False
        truncated = False

        # Perform the action
        if action == 0:  # Build 1 unit
            self.towers[curr] += 1
        elif action == 1:  # Build 2 units
            self.towers[curr] += 2
        elif action == 2:  # Build 4 units
            self.towers[curr] += 4
        elif action == 3:  # Attack opponent
            self.towers[opp] -= 3
            if self.towers[opp] < 0:
                self.towers[opp] = 0  # Ensure tower height doesn't go below zero

        # Check for overshooting the tower limit
        if self.towers[curr] > 15:
            self.done = True
            terminated = True
            reward = -10  # Loss for exceeding tower height
            return np.array(self._get_obs()), reward, terminated, truncated, {}

        # Check for winning condition
        if self.towers[curr] == 15:
            self.done = True
            terminated = True
            reward = 1  # Reward for winning the game
            return np.array(self._get_obs()), reward, terminated, truncated, {}

        # Check if opponent has already won
        if self.towers[opp] == 15:
            self.done = True
            terminated = True
            reward = -10  # Loss if opponent has already won
            return np.array(self._get_obs()), reward, terminated, truncated, {}

        # Switch to the next player
        self.current_player = opp

        return np.array(self._get_obs()), reward, terminated, truncated, {}

    def render(self):
        curr = self.current_player
        opp = 1 - curr
        return (
            f"Player {curr + 1}, it's your turn.\n"
            f"Your tower height: {self.towers[curr]}\n"
            f"Opponent's tower height: {self.towers[opp]}"
        )

    def valid_moves(self):
        curr = self.current_player
        opp = 1 - curr
        valid_actions = []

        # Check for valid building actions
        if self.towers[curr] + 1 <= 15:
            valid_actions.append(0)  # Build 1 unit
        if self.towers[curr] + 2 <= 15:
            valid_actions.append(1)  # Build 2 units
        if self.towers[curr] + 4 <= 15:
            valid_actions.append(2)  # Build 4 units

        # Check if attack is valid
        if self.towers[opp] >= 3:
            valid_actions.append(3)  # Attack opponent

        return valid_actions

    def _get_obs(self):
        # Observation is the current state from the current player's perspective
        curr = self.current_player
        opp = 1 - curr
        return [self.towers[curr], self.towers[opp]]
