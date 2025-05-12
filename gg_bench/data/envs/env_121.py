import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Action space: 10 positions * 5 numbers = 50 possible actions
        self.action_space = spaces.Discrete(50)
        # Observation space: 10 positions with numbers 0 (empty) to 5
        self.observation_space = spaces.Box(low=0, high=5, shape=(10,), dtype=np.int8)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_line = np.zeros(10, dtype=np.int8)
        self.current_player = 1  # Players are 1 and 2
        self.done = False
        return self.number_line.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return self.number_line.copy(), 0, True, False, {}

        # Decode action into position (0-9) and number (1-5)
        position_index = action // 5
        number = (action % 5) + 1

        # Check if move is valid
        if (
            position_index < 0
            or position_index >= 10
            or self.number_line[position_index] != 0
        ):
            # Invalid move
            self.done = True
            return self.number_line.copy(), -10, True, False, {}

        # Place the number on the Number Line
        self.number_line[position_index] = number

        # Resolve collisions
        collision_occurred = False
        positions_to_remove = set()
        # Check for adjacent identical numbers
        if (position_index > 0 and self.number_line[position_index - 1] == number) or (
            position_index < 9 and self.number_line[position_index + 1] == number
        ):
            collision_occurred = True
            positions_to_check = [position_index]
            positions_checked = set()
            # Begin collision resolution
            while positions_to_check:
                idx = positions_to_check.pop()
                positions_checked.add(idx)
                positions_to_remove.add(idx)
                # Check left
                if (
                    idx > 0
                    and self.number_line[idx - 1] == number
                    and (idx - 1) not in positions_checked
                ):
                    positions_to_check.append(idx - 1)
                # Check right
                if (
                    idx < 9
                    and self.number_line[idx + 1] == number
                    and (idx + 1) not in positions_checked
                ):
                    positions_to_check.append(idx + 1)
            # Remove collided numbers
            for idx in positions_to_remove:
                self.number_line[idx] = 0

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Current player wins as the opponent cannot make a valid move
            # Switch back to the winning player
            self.current_player = 2 if self.current_player == 1 else 1
            self.done = True
            return self.number_line.copy(), 1, True, False, {}

        # Return the observation, reward, termination status, truncation status, and info
        return self.number_line.copy(), 0, False, False, {}

    def render(self):
        number_line_str = "Number Line Positions: [1][2][3][4][5][6][7][8][9][10]\n"
        current_state_str = "Current State:         "
        for num in self.number_line:
            current_state_str += f"[{int(num) if num != 0 else ' '}]"
        current_state_str += "\n"
        return number_line_str + current_state_str

    def valid_moves(self):
        valid_actions = []
        for position_index in range(10):
            if self.number_line[position_index] == 0:
                for number in range(1, 6):
                    action = position_index * 5 + (number - 1)
                    valid_actions.append(action)
        return valid_actions
