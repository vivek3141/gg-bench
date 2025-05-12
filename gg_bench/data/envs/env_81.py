import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_sum=15):
        super(CustomEnv, self).__init__()

        # Target sum for winning condition
        self.target_sum = target_sum

        # Action space: integers from 0 to 8 corresponding to numbers 1 to 9
        self.action_space = spaces.Discrete(
            9
        )  # Actions are 0-8 representing numbers 1-9

        # Observation space: last three numbers, current player, target sum
        # last_three_numbers: values 0-9 (0 if not enough numbers yet)
        # current_player: -1 or 1
        # target_sum: integer value (e.g., 15)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0]),
            high=np.array([9, 9, 9, 1, 27]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the sequence and current player
        self.sequence = []
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        if action not in self.valid_moves():
            # Invalid move (should not happen as all moves are valid)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Convert action to number (actions 0-8 correspond to numbers 1-9)
        number = action + 1
        self.sequence.append(number)

        # Check for winning condition
        if len(self.sequence) >= 3:
            last_three_sum = sum(self.sequence[-3:])
            if last_three_sum == self.target_sum:
                # Current player wins
                self.done = True
                return self._get_observation(), 1, True, False, {}

        # If game not over, switch player and continue
        self.current_player *= -1
        return (
            self._get_observation(),
            -10,
            False,
            False,
            {},
        )  # Negative reward for valid move

    def render(self):
        # Visual representation of the sequence and current player
        sequence_str = ", ".join(map(str, self.sequence))
        return f"Sequence: [{sequence_str}]\nCurrent Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # All numbers 1-9 are valid moves at any time
        return list(range(9))  # Actions 0-8 correspond to numbers 1-9

    def _get_observation(self):
        # Prepare the observation array
        last_three_numbers = (
            [0, 0, 0]
            if len(self.sequence) == 0
            else (
                [0, 0] + self.sequence[-1:]
                if len(self.sequence) == 1
                else (
                    [0] + self.sequence[-2:]
                    if len(self.sequence) == 2
                    else self.sequence[-3:]
                )
            )
        )

        observation = np.array(
            [
                last_three_numbers[0],
                last_three_numbers[1],
                last_three_numbers[2],
                self.current_player,
                self.target_sum,
            ],
            dtype=np.int32,
        )

        return observation
