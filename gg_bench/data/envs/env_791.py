import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Enter bridge, 1 - Move forward 1 cell, 2 - Move forward 2 cells
        self.action_space = spaces.Discrete(3)
        # Observations: Positions of Player 1 and Player 2
        self.observation_space = spaces.Box(low=0, high=9, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize player positions: Player 1 at position 0, Player 2 at position 9
        self.positions = [0, 9]
        self.current_player = 1  # 1 for Player 1, 2 for Player 2
        self.done = False
        return np.array(self.positions), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array(self.positions), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return np.array(self.positions), -10, True, False, {}

        # Process the action
        current_idx = self.current_player - 1  # Player index (0 or 1)
        opponent_idx = 1 - current_idx

        # Off the bridge
        if self.positions[current_idx] == 0 or self.positions[current_idx] == 9:
            # Action must be 0 (Enter bridge)
            if action != 0:
                self.done = True
                return np.array(self.positions), -10, True, False, {}

            # Enter the bridge
            if self.current_player == 1:
                self.positions[current_idx] = 1
            else:
                self.positions[current_idx] = 8
        else:
            # On the bridge
            if action == 1:
                move = 1
            elif action == 2:
                move = 2
            else:
                self.done = True
                return np.array(self.positions), -10, True, False, {}

            # Move forward
            if self.current_player == 1:
                new_position = self.positions[current_idx] + move
                # Ensure not beyond cell 8
                if new_position > 8:
                    new_position = 8
            else:
                new_position = self.positions[current_idx] - move
                # Ensure not before cell 1
                if new_position < 1:
                    new_position = 1

            # Check for capture or passing
            opponent_position = self.positions[opponent_idx]

            if new_position == opponent_position:
                # Capture occurs: Current player wins
                self.positions[current_idx] = new_position
                self.done = True
                return np.array(self.positions), 1, True, False, {}
            elif (self.current_player == 1 and new_position > opponent_position) or (
                self.current_player == 2 and new_position < opponent_position
            ):
                # Passed opponent without capturing: Current player loses
                self.positions[current_idx] = new_position
                self.done = True
                return np.array(self.positions), -1, True, False, {}
            else:
                # Valid move
                self.positions[current_idx] = new_position

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1
        return (
            np.array(self.positions),
            0,
            self.done,
            False,
            {},
        )  # Observation, reward, done, info

    def render(self):
        bridge = ["_"] * 8  # Representation of bridge cells 1 to 8
        for i in range(2):
            pos = self.positions[i]
            if 1 <= pos <= 8:
                token = "P1" if i == 0 else "P2"
                bridge[pos - 1] = token
        bridge_str = " ".join(bridge)
        print(f"Bridge: {bridge_str}")
        print(f"Player positions: {self.positions}")
        print(f"Current player: {'P1' if self.current_player == 1 else 'P2'}")

    def valid_moves(self):
        if self.done:
            return []

        current_idx = self.current_player - 1  # Player index (0 or 1)
        position = self.positions[current_idx]
        if position == 0 or position == 9:
            return [0]  # Only action is to enter the bridge
        else:
            return [1, 2]  # Can move forward 1 or 2 cells
