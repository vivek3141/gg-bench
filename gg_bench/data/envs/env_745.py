import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers 1-9 represented as indices 0-8
        self.action_space = spaces.Discrete(9)

        # Observation space: 18 integers representing available numbers and shared sequence
        # First 9 positions: availability of numbers 1-9 (1 available, 0 taken)
        # Next 9 positions: shared sequence (0 if position is empty)
        self.observation_space = spaces.Box(low=0, high=9, shape=(18,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize available numbers (1 available, 0 taken)
        self.available_numbers = np.ones(9, dtype=np.int32)
        # Initialize shared sequence (0 indicates empty position)
        self.shared_sequence = np.zeros(9, dtype=np.int32)
        # Current player (1 or 2)
        self.current_player = 1
        # Position in the shared sequence
        self.turn_counter = 0
        # Game termination flag
        self.terminated = False

        observation = np.concatenate([self.available_numbers, self.shared_sequence])
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if action is valid (number is available)
        if self.terminated or self.available_numbers[action] == 0:
            # Invalid move
            observation = np.concatenate([self.available_numbers, self.shared_sequence])
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        number = action + 1  # Numbers 1-9
        # Add number to shared sequence
        self.shared_sequence[self.turn_counter] = number
        self.turn_counter += 1
        # Mark number as taken
        self.available_numbers[action] = 0

        # Check for victory
        sum_sequence = 0
        for i in range(self.turn_counter - 1, -1, -1):
            sum_sequence += self.shared_sequence[i]
            if sum_sequence == 15:
                # Winning sequence found
                observation = np.concatenate(
                    [self.available_numbers, self.shared_sequence]
                )
                return (
                    observation,
                    1,
                    True,
                    False,
                    {},
                )  # Observation, reward, terminated, truncated, info
            elif sum_sequence > 15:
                break  # No need to check further
        # Check if all numbers are used
        if np.all(self.available_numbers == 0):
            # According to the rules, the player who placed the last number loses
            observation = np.concatenate([self.available_numbers, self.shared_sequence])
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Switch players
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Game continues
        observation = np.concatenate([self.available_numbers, self.shared_sequence])
        return (
            observation,
            -10,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Render the game state as a string
        available_nums = [
            str(i + 1) for i in range(9) if self.available_numbers[i] == 1
        ]
        shared_seq = [str(num) for num in self.shared_sequence if num > 0]
        render_str = "Available Numbers: " + ", ".join(available_nums) + "\n"
        render_str += "Shared Sequence: " + ", ".join(shared_seq) + "\n"
        render_str += f"Current Player: Player {self.current_player}\n"
        return render_str

    def valid_moves(self):
        # Return indices of available numbers
        return [i for i in range(9) if self.available_numbers[i] == 1]
