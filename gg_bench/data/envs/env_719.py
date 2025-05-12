import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 flip actions + 56 swap actions = 64 total actions
        self.action_space = spaces.Discrete(64)

        # Define observation space: Target pattern + current player's binary string + opponent's binary string
        # Each consists of 8 bits, so total 24 bits
        self.observation_space = spaces.MultiBinary(24)

        # Mapping from action indices to operations
        self.action_to_operation = {}
        # Flip actions (indices 0-7)
        for action_index in range(8):
            position = action_index  # Positions 0-7
            self.action_to_operation[action_index] = ("flip", position)
        # Swap actions (indices 8-63)
        action_number = 0
        for i in range(8):
            for j in range(8):
                if i != j:
                    action_index = 8 + action_number
                    self.action_to_operation[action_index] = ("swap", i, j)
                    action_number += 1

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate random target pattern
        self.target_pattern = np.random.randint(0, 2, 8, dtype=np.int8)

        # Initialize player binary strings to zeros
        self.player_binaries = {
            1: np.zeros(8, dtype=np.int8),
            2: np.zeros(8, dtype=np.int8),
        }

        # Set current player
        self.current_player = 1

        # Game not done
        self.done = False

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}
        if not self.action_space.contains(action):
            self.done = True
            return self._get_observation(), -10, True, False, {}

        operation = self.action_to_operation[action]

        # Perform the operation
        player_binary = self.player_binaries[self.current_player]
        if operation[0] == "flip":
            position = operation[1]
            # Flip the bit at position
            player_binary[position] = 1 - player_binary[position]
        elif operation[0] == "swap":
            pos1, pos2 = operation[1], operation[2]
            # Swap bits at pos1 and pos2
            player_binary[pos1], player_binary[pos2] = (
                player_binary[pos2],
                player_binary[pos1],
            )

        # Update player's binary string
        self.player_binaries[self.current_player] = player_binary

        # Check for victory
        if np.array_equal(player_binary, self.target_pattern):
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, {}

        # Switch player
        self.current_player = self._opponent_player()

        # Return observation for the next player
        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        target_str = " ".join(map(str, self.target_pattern))
        player1_str = " ".join(map(str, self.player_binaries[1]))
        player2_str = " ".join(map(str, self.player_binaries[2]))
        current_player = self.current_player

        render_str = f"Target Pattern:\n{target_str}\n\n"
        render_str += f"Player 1 Binary String:\n{player1_str}\n"
        render_str += f"Player 2 Binary String:\n{player2_str}\n\n"
        render_str += f"Current Player: Player {current_player}\n"
        return render_str

    def valid_moves(self):
        # All actions from 0 to 63 are valid
        return list(range(64))

    def _get_observation(self):
        # Construct observation: target pattern + current player's binary string + opponent's binary string
        opponent_player = self._opponent_player()
        observation = np.concatenate(
            (
                self.target_pattern,
                self.player_binaries[self.current_player],
                self.player_binaries[opponent_player],
            )
        )
        return observation

    def _opponent_player(self):
        return 2 if self.current_player == 1 else 1
