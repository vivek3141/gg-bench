import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space

        # Action space is Discrete(8), actions 0-3 flip own bits at position 0-3
        # Actions 4-7 reset opponent's bits at position 0-3
        self.action_space = spaces.Discrete(8)

        # Observation space is the bits of both players and current player indicator
        # We'll define observation as an array of 9 integers:
        # Positions 0-3: bits of player 0
        # Positions 4-7: bits of player 1
        # Position 8: current player (0 or 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_bits = np.zeros((2, 4), dtype=np.int8)  # Two players, 4 bits each
        self.current_player = 0  # Player 0 starts
        self.done = False
        self.first_move = True  # First move of the game
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            # The game is over
            return self._get_obs(), -10, True, False, {}

        # Decode action
        player = self.current_player
        opponent = 1 - self.current_player

        if action >= 0 and action <= 3:
            # Flip own bit at position (action)
            bit_pos = action
            valid = self._can_flip(player, bit_pos)
            if not valid:
                # Invalid action
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                self.player_bits[player][bit_pos] = 1
        elif action >= 4 and action <= 7:
            # Reset opponent's bit at position (action - 4)
            bit_pos = action - 4
            valid = self._can_reset(opponent, bit_pos)
            if not valid:
                # Invalid action
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                self.player_bits[opponent][bit_pos] = 0
        else:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check if current player wins
        if np.all(self.player_bits[player] == 1):
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch player
        self.current_player = opponent
        self.first_move = False
        return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        # Observation is an array of 9 elements
        obs = np.zeros(9, dtype=np.int8)
        obs[0:4] = self.player_bits[0]
        obs[4:8] = self.player_bits[1]
        obs[8] = self.current_player
        return obs

    def _can_flip(self, player, bit_pos):
        # Check if the player can flip their own bit at bit_pos
        if self.player_bits[player][bit_pos] == 1:
            return False  # Bit already 1

        if self.first_move and self.current_player == 0:
            # First move exception for Player 0's first move
            return True

        # Check adjacency
        adjacent_indices = []
        if bit_pos == 0:
            adjacent_indices = [1]
        elif bit_pos == 1:
            adjacent_indices = [0, 2]
        elif bit_pos == 2:
            adjacent_indices = [1, 3]
        elif bit_pos == 3:
            adjacent_indices = [2]
        else:
            return False  # Invalid bit position

        for adj in adjacent_indices:
            if self.player_bits[player][adj] == 1:
                return True
        return False

    def _can_reset(self, opponent, bit_pos):
        # Check if the current player can reset opponent's bit at bit_pos
        if self.player_bits[opponent][bit_pos] == 0:
            return False  # Bit already 0

        # Check adjacent bits
        adjacent_indices = []
        if bit_pos == 0:
            adjacent_indices = [1]
        elif bit_pos == 1:
            adjacent_indices = [0, 2]
        elif bit_pos == 2:
            adjacent_indices = [1, 3]
        elif bit_pos == 3:
            adjacent_indices = [2]
        else:
            return False  # Invalid bit position

        adjacent_bits = [self.player_bits[opponent][i] for i in adjacent_indices]

        if len(adjacent_bits) == 1:
            if adjacent_bits[0] == 1:
                # Cannot reset if adjacent bit is 1 and only one neighbor
                return False
            else:
                return True
        else:
            if adjacent_bits[0] == 1 and adjacent_bits[1] == 1:
                return False  # Cannot reset if both adjacent bits are 1
            else:
                return True
        return True

    def render(self):
        player0_bits = " ".join(str(bit) for bit in self.player_bits[0])
        player1_bits = " ".join(str(bit) for bit in self.player_bits[1])
        output = f"Player 0 bits: {player0_bits}\n"
        output += f"Player 1 bits: {player1_bits}\n"
        output += f"Current player: {self.current_player}"
        return output

    def valid_moves(self):
        # Return a list of valid moves for the current player
        moves = []
        player = self.current_player
        opponent = 1 - player

        # Own bits flip actions
        for i in range(4):
            action = i
            if self._can_flip(player, i):
                moves.append(action)

        # Reset opponent's bits actions
        for i in range(4):
            action = i + 4
            if self._can_reset(opponent, i):
                moves.append(action)
        return moves
