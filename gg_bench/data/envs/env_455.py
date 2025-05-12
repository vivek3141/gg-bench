import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_NUMBER = 16  # Starting number
        self.MAX_NUMBERS = 20  # Max length of the Number List
        self.MAX_SPLITS = self.MAX_NUMBER - 3  # Max possible splits for any number
        self.MAX_ACTIONS = self.MAX_NUMBERS * self.MAX_SPLITS

        # Define action and observation space
        self.action_space = spaces.Discrete(self.MAX_ACTIONS)

        # Observation space: Number List with fixed size, padded with zeros
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_NUMBER, shape=(self.MAX_NUMBERS,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize Number List with the starting number and pad with zeros
        self.NumberList = [self.MAX_NUMBER] + [0] * (self.MAX_NUMBERS - 1)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array(self.NumberList, dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(self.NumberList, dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Decode action into number index and split value
        number_index = action // self.MAX_SPLITS
        split_k = (action % self.MAX_SPLITS) + 2  # k ranges from 2 to MAX_NUMBER - 2

        # Get the valid numbers in the Number List
        numbers_in_list = [n for n in self.NumberList if n != 0]

        # Check if number index is valid
        if number_index >= len(numbers_in_list):
            self.done = True
            return np.array(self.NumberList, dtype=np.int32), -10, True, False, {}

        N = numbers_in_list[number_index]

        # Check if the selected number can be divided
        if N <= 2 or split_k < 2 or split_k > N - 2 or N - split_k < 2:
            self.done = True
            return np.array(self.NumberList, dtype=np.int32), -10, True, False, {}

        # Remove the original number from the Number List
        idx_in_NumberList = self.NumberList.index(N)
        self.NumberList[idx_in_NumberList] = 0

        # Add the two new numbers to the Number List
        added = False
        for i in range(len(self.NumberList)):
            if self.NumberList[i] == 0:
                self.NumberList[i] = split_k
                added = True
                break
        if not added:
            # No space to add new number
            self.done = True
            return np.array(self.NumberList, dtype=np.int32), -10, True, False, {}
        added = False
        for i in range(len(self.NumberList)):
            if self.NumberList[i] == 0:
                self.NumberList[i] = N - split_k
                added = True
                break
        if not added:
            # No space to add new number
            self.done = True
            return np.array(self.NumberList, dtype=np.int32), -10, True, False, {}

        # Clean up the Number List: remove zeros and pad to fixed size
        self.NumberList = [n for n in self.NumberList if n != 0]
        self.NumberList.sort(reverse=True)
        self.NumberList += [0] * (self.MAX_NUMBERS - len(self.NumberList))

        # Check if there are valid moves for the next player
        valid_moves_next = self.valid_moves()
        if not valid_moves_next:
            # Current player wins
            self.done = True
            return np.array(self.NumberList, dtype=np.int32), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        return np.array(self.NumberList, dtype=np.int32), 0, False, False, {}

    def render(self):
        numbers_in_list = [n for n in self.NumberList if n != 0]
        render_str = f"Number List: {numbers_in_list}\n"
        render_str += f"Player {1 if self.current_player == 1 else 2}'s Turn"
        return render_str

    def valid_moves(self):
        valid_moves = []
        numbers_in_list = [n for n in self.NumberList if n != 0]
        for num_idx, N in enumerate(numbers_in_list):
            if N > 2:
                for k in range(2, N - 1):
                    if N - k >= 2:
                        action = num_idx * self.MAX_SPLITS + (k - 2)
                        valid_moves.append(action)
        return valid_moves
