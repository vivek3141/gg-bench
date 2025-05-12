import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Choose a number from 1 to 9 (actions 0 to 8 correspond to numbers 1 to 9)
        self.action_space = spaces.Discrete(9)

        # Observation space: Both players' pyramids (12 positions total)
        # Each position can be 0 (empty) or a number from 1 to 9
        self.observation_space = spaces.Box(low=0, high=9, shape=(12,), dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player's pyramid: Positions 0-5 for Player 1, 6-11 for Player 2
        self.state = np.zeros(12, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.state.copy(), 0, True, False, {}

        number = action + 1  # Convert action to number (1 to 9)

        # Determine current player's indices
        player_offset = 0 if self.current_player == 1 else 6
        pyramid = self.state[player_offset : player_offset + 6]

        # Find next available position
        next_pos = self._next_available_position(pyramid)

        if next_pos == -1:
            # No positions available (should not happen in normal gameplay)
            self.done = True
            reward = 1 if self._check_win(pyramid) else 0
            return self.state.copy(), reward, True, False, {}

        # Determine valid numbers for the next position
        valid_numbers = self._valid_numbers_for_position(pyramid, next_pos)

        if number in valid_numbers:
            # Place the number
            pyramid[next_pos] = number
            self.state[player_offset : player_offset + 6] = pyramid

            # Check for win
            if self._check_win(pyramid):
                self.done = True
                return self.state.copy(), 1, True, False, {}

            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1
            return self.state.copy(), 0, False, False, {}
        else:
            # Invalid move
            self.done = True
            return self.state.copy(), -10, True, False, {}

    def render(self):
        p1_pyramid = self.state[0:6]
        p2_pyramid = self.state[6:12]
        render_str = ""
        render_str += "Player 1's Pyramid:\n"
        render_str += self._render_pyramid(p1_pyramid)
        render_str += "\nPlayer 2's Pyramid:\n"
        render_str += self._render_pyramid(p2_pyramid)
        return render_str

    def valid_moves(self):
        # Returns a list of valid actions (indices 0-8 corresponding to numbers 1-9)
        player_offset = 0 if self.current_player == 1 else 6
        pyramid = self.state[player_offset : player_offset + 6]

        next_pos = self._next_available_position(pyramid)

        if next_pos == -1:
            return []  # No valid moves; pyramid is complete

        valid_numbers = self._valid_numbers_for_position(pyramid, next_pos)
        # Convert valid numbers to action indices (number - 1)
        return [n - 1 for n in valid_numbers]

    # Helper functions
    def _next_available_position(self, pyramid):
        for i in range(6):
            if pyramid[i] == 0:
                return i
        return -1  # No available positions

    def _valid_numbers_for_position(self, pyramid, position):
        # Bottom level positions (indices 0-2)
        if position <= 2:
            return list(range(1, 10))  # Numbers 1 to 9
        elif position == 3:
            below_left = pyramid[0]
            below_right = pyramid[1]
        elif position == 4:
            below_left = pyramid[1]
            below_right = pyramid[2]
        elif position == 5:
            below_left = pyramid[3]
            below_right = pyramid[4]
        else:
            return []  # Invalid position

        # If the positions below are not filled, no valid moves
        if below_left == 0 or below_right == 0:
            return []

        possible_numbers = set()
        sum_value = below_left + below_right
        diff_value = abs(below_left - below_right)

        if 1 <= sum_value <= 9:
            possible_numbers.add(sum_value)
        if 1 <= diff_value <= 9:
            possible_numbers.add(diff_value)

        return list(possible_numbers)

    def _check_win(self, pyramid):
        # Player wins if pyramid is fully filled
        return all(pyramid[i] != 0 for i in range(6))

    def _render_pyramid(self, pyramid):
        lines = []
        lines.append(f"    [{pyramid[5] if pyramid[5] !=0 else ' '}]    ")
        lines.append(
            f"  [{pyramid[3] if pyramid[3] !=0 else ' '}][{pyramid[4] if pyramid[4] !=0 else ' '}]  "
        )
        lines.append(
            f"[{pyramid[0] if pyramid[0] !=0 else ' '}][{pyramid[1] if pyramid[1] !=0 else ' '}][{pyramid[2] if pyramid[2] !=0 else ' '}]"
        )
        return "\n".join(lines)
