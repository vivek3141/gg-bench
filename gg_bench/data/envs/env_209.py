import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Light Attack, 1 - Heavy Attack, 2 - Fortify
        self.action_space = spaces.Discrete(3)

        # Define observation space: [own_barrier, opponent_barrier], barriers range from 0 to 10
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Initialize game state
        self.player1_barrier = 10
        self.player2_barrier = 10
        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_barrier = 10
        self.player2_barrier = 10
        self.current_player = 1
        self.done = False
        return self.current_observation(), {}  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self.current_observation(), 0, True, False, {}

        # Check if action is valid
        if action not in self.valid_moves():
            self.done = True
            return self.current_observation(), -10, True, False, {}

        # Get current barriers
        if self.current_player == 1:
            own_barrier = self.player1_barrier
            opp_barrier = self.player2_barrier
        else:
            own_barrier = self.player2_barrier
            opp_barrier = self.player1_barrier

        # Apply action
        if action == 0:  # Light Attack
            opp_barrier -= 2
        elif action == 1:  # Heavy Attack
            opp_barrier -= 3
        elif action == 2:  # Fortify
            own_barrier += 1

        # Enforce barrier limits
        own_barrier = min(max(own_barrier, 0), 10)
        opp_barrier = min(max(opp_barrier, 0), 10)

        # Update barriers
        if self.current_player == 1:
            self.player1_barrier = own_barrier
            self.player2_barrier = opp_barrier
        else:
            self.player2_barrier = own_barrier
            self.player1_barrier = opp_barrier

        # Check if opponent's barrier has reached zero
        if opp_barrier == 0:
            self.done = True
            return self.current_observation(), 1, True, False, {}

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return observation
        return self.current_observation(), 0, False, False, {}

    def render(self):
        return f"Player 1 Barrier: {self.player1_barrier}\nPlayer 2 Barrier: {self.player2_barrier}"

    def valid_moves(self):
        if self.done:
            return []
        own_barrier = (
            self.player1_barrier if self.current_player == 1 else self.player2_barrier
        )
        valid_actions = [0, 2]  # Light Attack and Fortify are always valid
        if own_barrier > 3:
            valid_actions.append(1)  # Heavy Attack is valid
        return valid_actions

    def current_observation(self):
        if self.current_player == 1:
            return np.array(
                [self.player1_barrier, self.player2_barrier], dtype=np.int32
            )
        else:
            return np.array(
                [self.player2_barrier, self.player1_barrier], dtype=np.int32
            )
