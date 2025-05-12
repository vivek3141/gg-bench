import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Numbers from 2 to 20 (indices 0 to 18)
        self.action_space = spaces.Discrete(19)

        # Define observation space
        # First 19 entries: -1 (opponent), 0 (available), 1 (current player)
        # Last entry: -1 (no last number) to 18 (index of last picked number)
        low = np.array([-1] * 19 + [-1], dtype=np.int8)
        high = np.array([1] * 19 + [18], dtype=np.int8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int8)

        # Internal variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.number_pool = np.zeros(
            19, dtype=np.int8
        )  # 0: available, 1: picked by Player 1, -1: picked by Player 2
        self.last_picked_number_index = [
            -1,
            -1,
        ]  # Indices of last numbers picked by players [-1 if none]
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}  # Game already over

        # Map current_player to index 0 (Player 1) or 1 (Player 2)
        current_player_index = 0 if self.current_player == 1 else 1
        opponent_player_index = 1 - current_player_index

        # Check if action is within valid range
        if action < 0 or action >= 19:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if number is available
        if self.number_pool[action] != 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if move is valid according to game rules
        # Get last picked number for current player
        last_picked_index = self.last_picked_number_index[current_player_index]
        selected_number = action + 2  # Map index to number (2 to 20)
        valid = False
        if last_picked_index == -1:
            # First turn for current player, any number is valid
            valid = True
        else:
            last_number = last_picked_index + 2
            if self._has_common_factor_greater_than_one(selected_number, last_number):
                valid = True

        if not valid:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Valid move, update game state
        self.number_pool[action] = self.current_player
        self.last_picked_number_index[current_player_index] = action

        # Check if opponent has valid moves
        opponent_valid_moves = self._get_valid_moves(self.current_player * -1)
        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch current player
            self.current_player *= -1
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        number_pool_str = "Available Numbers:\n"
        for i in range(19):
            if self.number_pool[i] == 0:
                number_pool_str += f"{i+2},"
        number_pool_str = number_pool_str.rstrip(",")

        p1_numbers = [i + 2 for i in range(19) if self.number_pool[i] == 1]
        p2_numbers = [i + 2 for i in range(19) if self.number_pool[i] == -1]

        p1_collection_str = f"Player 1's Collection: {p1_numbers}"
        p2_collection_str = f"Player 2's Collection: {p2_numbers}"

        render_str = f"{number_pool_str}\n{p1_collection_str}\n{p2_collection_str}"
        return render_str

    def valid_moves(self):
        return self._get_valid_moves(self.current_player)

    def _get_valid_moves(self, player):
        player_index = 0 if player == 1 else 1
        valid_moves = []

        if self.last_picked_number_index[player_index] == -1:
            # First turn for this player, any available number is valid
            valid_moves = [i for i in range(19) if self.number_pool[i] == 0]
        else:
            last_number = (
                self.last_picked_number_index[player_index] + 2
            )  # Map index to number
            for i in range(19):
                if self.number_pool[i] == 0:
                    current_number = i + 2
                    if self._has_common_factor_greater_than_one(
                        current_number, last_number
                    ):
                        valid_moves.append(i)
        return valid_moves

    def _has_common_factor_greater_than_one(self, num1, num2):
        # Return True if num1 and num2 share a common factor greater than 1
        min_num = min(num1, num2)
        for i in range(2, min_num + 1):
            if num1 % i == 0 and num2 % i == 0:
                return True
        return False

    def _get_observation(self):
        # Observation: array of length 20
        # Indices 0-18: number pool state (-1, 0, 1)
        # Index 19: last picked number index (-1 to 18)
        current_player_index = 0 if self.current_player == 1 else 1
        last_picked_index = self.last_picked_number_index[current_player_index]
        observation = np.append(self.number_pool.copy(), last_picked_index).astype(
            np.int8
        )
        return observation
