import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0: Move forward 1
        # 1: Move forward 2
        # 2: Move backward 1
        # 3: Move backward 2
        self.action_space = spaces.Discrete(4)

        # Observation space: Positions of both players on the number line
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.array(
            [0, 10], dtype=np.int8
        )  # Player 1 at position 0, Player 2 at position 10
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.positions, {}

    def step(self, action):
        if self.done:
            return self.positions, 0, True, False, {}

        # Map action to movement delta based on current player
        delta_pos = self._map_action_to_delta(action)

        current_pos = self.positions[self.current_player - 1]
        new_pos = current_pos + delta_pos

        # Check if move is valid (within number line boundaries)
        if new_pos < 0 or new_pos > 10:
            self.done = True
            return self.positions, -10, True, False, {}

        # Update player's position
        self.positions[self.current_player - 1] = new_pos

        # Check for win conditions
        opponent_player = 2 if self.current_player == 1 else 1
        opponent_pos = self.positions[opponent_player - 1]
        opponent_base = 0 if opponent_player == 1 else 10

        if new_pos == opponent_pos:
            # Capture opponent's token
            self.done = True
            return self.positions, 1, True, False, {}
        elif new_pos == opponent_base:
            # Reached opponent's base
            self.done = True
            return self.positions, 1, True, False, {}

        # Switch to the next player
        self.current_player = opponent_player

        return self.positions, 0, False, False, {}

    def render(self):
        # Create a string representation of the number line with player positions
        visualization = ""
        for pos in range(11):
            if pos == self.positions[0] and pos == self.positions[1]:
                visualization += f"{pos}[A/B] "
            elif pos == self.positions[0]:
                visualization += f"{pos}[A] "
            elif pos == self.positions[1]:
                visualization += f"{pos}[B] "
            else:
                visualization += f"{pos} "
        return visualization.strip()

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        current_pos = self.positions[self.current_player - 1]
        for action in range(4):
            delta_pos = self._map_action_to_delta(action)
            new_pos = current_pos + delta_pos
            if 0 <= new_pos <= 10:
                valid_actions.append(action)
        return valid_actions

    def _map_action_to_delta(self, action):
        # Map action indices to movement deltas based on the current player
        if self.current_player == 1:
            mapping = {
                0: +1,  # Move forward 1
                1: +2,  # Move forward 2
                2: -1,  # Move backward 1
                3: -2,  # Move backward 2
            }
        else:  # Player 2
            mapping = {
                0: -1,  # Move forward 1 (towards Player 1's base)
                1: -2,  # Move forward 2
                2: +1,  # Move backward 1
                3: +2,  # Move backward 2
            }
        return mapping[action]
