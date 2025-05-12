import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 36 possible actions (flip 1 or 2 bits among bits 1-8)
        self.action_space = spaces.Discrete(36)

        # Define observation space: shared binary number and target binary number (16 bits total)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared binary number to 00000000
        self.shared_binary = np.zeros(8, dtype=np.int8)
        # Generate a random target binary number
        self.target_binary = np.random.randint(0, 2, size=8, dtype=np.int8)
        # Keep track of last action per player to enforce move restrictions
        self.last_actions = {1: None, -1: None}
        # Current player (1 or -1)
        self.current_player = 1
        # Game over flag
        self.done = False

        # Generate action map: maps action indices to bits to flip
        self.action_map = []
        # Actions for flipping one bit
        for i in range(8):
            self.action_map.append((i,))
        # Actions for flipping two bits
        for i in range(8):
            for j in range(i + 1, 8):
                self.action_map.append((i, j))

        observation = np.concatenate([self.shared_binary, self.target_binary])
        return observation, {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Validate action
        if action < 0 or action >= len(self.action_map):
            # Invalid action index
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Get bits to flip from action map
        bits_to_flip = self.action_map[action]

        # Check if action is repeating the previous move
        if self.last_actions[self.current_player] == bits_to_flip:
            # Invalid move: repeating previous move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Flip the bits and check for changes
        new_shared_binary = self.shared_binary.copy()
        for bit in bits_to_flip:
            # Flip the bit
            new_shared_binary[bit] ^= 1

        # Check if the shared binary number has changed
        if np.array_equal(new_shared_binary, self.shared_binary):
            # Invalid move: no change in shared binary number
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Update the shared binary number
        self.shared_binary = new_shared_binary
        # Update last action for the current player
        self.last_actions[self.current_player] = bits_to_flip

        # Check for win condition
        if np.array_equal(self.shared_binary, self.target_binary):
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        reward = 0  # No reward for regular move

        return self._get_obs(), reward, False, False, {}

    def render(self):
        shared_str = "".join(map(str, self.shared_binary))
        target_str = "".join(map(str, self.target_binary))
        return (
            f"Shared Binary Number: {shared_str}\nTarget Binary Number: {target_str}\n"
        )

    def valid_moves(self):
        valid_actions = []
        for action, bits in enumerate(self.action_map):
            # Check if action is not repeating the previous move
            if self.last_actions[self.current_player] == bits:
                continue
            else:
                valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Combine shared binary number and target binary number into observation
        observation = np.concatenate([self.shared_binary, self.target_binary])
        return observation
