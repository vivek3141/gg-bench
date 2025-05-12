import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space: swap adjacent positions, actions 0 to 3 corresponding to swaps between positions:
        # 0&1, 1&2, 2&3, 3&4
        self.action_space = spaces.Discrete(4)

        # Observation space: sequence of numbers from 1 to 5
        self.observation_space = spaces.Box(low=1, high=5, shape=(5,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Start with a random permutation of [1,2,3,4,5]
        self.sequence = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.np_random.shuffle(self.sequence)

        self.current_player = 1  # Can be 1 or -1 to represent player 1 or 2
        self.done = False

        return self.sequence.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                self.sequence.copy(),
                0,
                True,
                False,
                {"error": "Game is already over"},
            )

        # Validate the action
        if action not in [0, 1, 2, 3]:
            # Invalid action index
            reward = -10
            self.done = True
            info = {"error": "Invalid action"}
            return self.sequence.copy(), reward, True, False, info

        # Swap the two numbers at positions action and action + 1
        pos1 = action
        pos2 = action + 1
        self.sequence[pos1], self.sequence[pos2] = (
            self.sequence[pos2],
            self.sequence[pos1],
        )

        # Check if sequence is in ascending order
        if np.all(self.sequence[:-1] <= self.sequence[1:]):
            # Current player wins
            reward = 1
            self.done = True
            return self.sequence.copy(), reward, True, False, {}
        else:
            # Game continues
            # Switch to next player
            self.current_player *= -1
            reward = 0
            return self.sequence.copy(), reward, False, False, {}

    def render(self):
        # Return a string representation of the current sequence
        position_str = "Position: " + " ".join(str(i + 1) for i in range(5))
        number_str = "Numbers:  " + " ".join(str(num) for num in self.sequence)
        return position_str + "\n" + number_str + "\n"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1, 2, 3]
