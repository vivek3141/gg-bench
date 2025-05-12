import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Actions:
        # 0-3: Flip own number, bit positions 1-4 (indices 0-3)
        # 4-7: Flip opponent's number, bit positions 1-4 (indices 0-3)
        self.action_space = spaces.Discrete(8)

        # Observation space: 12 bits
        # Player 1's binary (bits 0-3)
        # Player 2's binary (bits 4-7)
        # Target binary (bits 8-11)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Initialize player binaries: arrays of zeros
        self.player_binaries = {
            1: np.zeros(4, dtype=np.int8),
            2: np.zeros(4, dtype=np.int8),
        }
        # Randomly set target binary number
        self.target_binary = self.np_random.integers(0, 2, size=4, dtype=np.int8)

        self.current_player = 1  # Player 1 starts
        self.done = False

        # Return the observation
        observation = self._get_observation()
        return observation, {}  # observation, info

    def _get_observation(self):
        # Construct the observation array of length 12
        obs = np.concatenate(
            (self.player_binaries[1], self.player_binaries[2], self.target_binary)
        )
        return obs

    def step(self, action):
        if self.done:
            # Invalid move: game has ended
            observation = self._get_observation()
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # reward, terminated, truncated, info

        if not self.action_space.contains(action):
            # Invalid action index
            observation = self._get_observation()
            self.done = True
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # reward, terminated, truncated, info

        # Determine the binary number to modify and bit position
        if action < 4:
            # Flip own number
            binary_to_modify = self.player_binaries[self.current_player]
        else:
            # Flip opponent's number
            opponent = 1 if self.current_player == 2 else 2
            binary_to_modify = self.player_binaries[opponent]

        bit_position = action % 4  # position 0 to 3 (bit positions 1 to 4)

        # Flip the bit
        binary_to_modify[bit_position] = 1 - binary_to_modify[bit_position]

        # Check for victory
        if np.array_equal(
            self.player_binaries[self.current_player], self.target_binary
        ):
            # Current player wins
            self.done = True
            observation = self._get_observation()
            return (
                observation,
                1,
                True,
                False,
                {},
            )  # reward, terminated, truncated, info

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2

        observation = self._get_observation()
        return observation, 0, False, False, {}  # reward, terminated, truncated, info

    def render(self):
        player1_binary_str = "".join(str(bit) for bit in self.player_binaries[1])
        player2_binary_str = "".join(str(bit) for bit in self.player_binaries[2])
        target_binary_str = "".join(str(bit) for bit in self.target_binary)
        return (
            f"Player 1's binary: {player1_binary_str}\n"
            f"Player 2's binary: {player2_binary_str}\n"
            f"Target binary:    {target_binary_str}\n"
            f"Current player: Player {self.current_player}"
        )

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(8))
