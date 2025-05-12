import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # There are 4 possible actions: L1, L2, R1, R2
        self.action_space = spaces.Discrete(4)

        # Observation: positions of both players (current player, opponent)
        # Each position ranges from 1 to 11
        self.observation_space = spaces.Box(low=1, high=11, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Both players start at position 6
        self.player_positions = {1: 6, 2: 6}
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Observation is [current player's position, opponent's position]
        return np.array(
            [
                self.player_positions[self.current_player],
                self.player_positions[3 - self.current_player],
            ],
            dtype=np.int32,
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if current player has valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # No valid moves; current player loses
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Map action to move direction and steps
        action_mapping = {0: ("L", 1), 1: ("L", 2), 2: ("R", 1), 3: ("R", 2)}
        direction, steps = action_mapping[action]
        current_position = self.player_positions[self.current_player]
        opponent_position = self.player_positions[3 - self.current_player]

        # Compute new position
        if direction == "L":
            new_position = current_position - steps
        else:  # direction == 'R'
            new_position = current_position + steps

        # Check boundaries
        if new_position < 1 or new_position > 11:
            # Move goes beyond boundaries; invalid move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Check if moving onto or passing over opponent
        positions_to_traverse = range(
            min(current_position, new_position), max(current_position, new_position) + 1
        )
        if opponent_position in positions_to_traverse:
            # Cannot move onto or pass over opponent; invalid move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Move is valid; update current player's position
        self.player_positions[self.current_player] = new_position

        # Check for victory condition
        if new_position == 1 or new_position == 11:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}

        # Switch to next player
        self.current_player = 3 - self.current_player
        reward = 0
        return self._get_obs(), reward, False, False, {}

    def render(self):
        # Create visual representation of the number line
        number_line = [" "] * 11  # Index 0 corresponds to position 1
        pos1 = self.player_positions[1] - 1  # Convert to 0-based index
        pos2 = self.player_positions[2] - 1

        if pos1 == pos2:
            number_line[pos1] = "P1/P2"
        else:
            number_line[pos1] = "P1"
            number_line[pos2] = "P2"

        line_str = "Number Line: " + " ".join(f"{i + 1:2d}" for i in range(11)) + "\n"
        positions_str = "             " + " ".join(
            f"{number_line[i]:>5}" if number_line[i] != " " else "     "
            for i in range(11)
        )
        return line_str + positions_str

    def valid_moves(self):
        valid_actions = []
        action_mapping = {0: ("L", 1), 1: ("L", 2), 2: ("R", 1), 3: ("R", 2)}
        current_position = self.player_positions[self.current_player]
        opponent_position = self.player_positions[3 - self.current_player]

        for action in range(4):
            direction, steps = action_mapping[action]
            # Compute new position
            if direction == "L":
                new_position = current_position - steps
            else:  # direction == 'R'
                new_position = current_position + steps

            # Check boundaries
            if new_position < 1 or new_position > 11:
                continue  # Move goes beyond boundaries

            # Check if moving onto or passing over opponent
            positions_to_traverse = range(
                min(current_position, new_position),
                max(current_position, new_position) + 1,
            )
            if opponent_position in positions_to_traverse:
                continue  # Cannot move onto or pass over opponent

            # Move is valid
            valid_actions.append(action)

        return valid_actions
