import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_N_PILES = 50
        self.MAX_PILE_SIZE = 100

        self.MAX_POSSIBLE_SPLITS_PER_PILE = self.MAX_PILE_SIZE // 2

        # Define action space (Discrete)
        self.action_space = spaces.Discrete(
            self.MAX_N_PILES * self.MAX_POSSIBLE_SPLITS_PER_PILE
        )

        # Define observation space (Box)
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_PILE_SIZE, shape=(self.MAX_N_PILES,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the piles
        self.piles = np.zeros(self.MAX_N_PILES, dtype=np.int32)
        self.piles[0] = 13  # Starting with default prime number of stones
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        return self.piles.copy()  # Return a copy of the piles array

    def _get_possible_splits(self, pile_size):
        splits = []
        for x in range(1, pile_size):
            y = pile_size - x
            if x != y and x > 0 and y > 0:
                splits.append((min(x, y), max(x, y)))  # Ensure x <= y
        # Remove duplicates
        splits = list(set(splits))
        splits.sort()
        return splits

    def _get_split_index(self, pile_size, x, y):
        splits = self._get_possible_splits(pile_size)
        try:
            index = splits.index((x, y))
        except ValueError:
            index = -1  # Invalid split
        return index

    def _get_action_id(self, pile_index, x, y):
        split_index = self._get_split_index(self.piles[pile_index], x, y)
        action_id = pile_index * self.MAX_POSSIBLE_SPLITS_PER_PILE + split_index
        return action_id

    def _get_pile_and_split_from_action(self, action_id):
        pile_index = action_id // self.MAX_POSSIBLE_SPLITS_PER_PILE
        split_index = action_id % self.MAX_POSSIBLE_SPLITS_PER_PILE

        if pile_index >= self.MAX_N_PILES or self.piles[pile_index] == 0:
            return None, None, None  # Invalid action

        pile_size = self.piles[pile_index]
        splits = self._get_possible_splits(pile_size)
        if split_index >= len(splits):
            return None, None, None  # Invalid action
        x, y = splits[split_index]
        return pile_index, x, y

    def valid_moves(self):
        valid_actions = []
        for pile_index in range(self.MAX_N_PILES):
            pile_size = self.piles[pile_index]
            if pile_size >= 3:
                splits = self._get_possible_splits(pile_size)
                for idx, split in enumerate(splits):
                    x, y = split
                    action_id = pile_index * self.MAX_POSSIBLE_SPLITS_PER_PILE + idx
                    valid_actions.append(action_id)
        return valid_actions

    def step(self, action):
        # If game is over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Get pile_index, x, y from action
        pile_index, x, y = self._get_pile_and_split_from_action(action)

        if pile_index is None or self.piles[pile_index] == 0:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        pile_size = self.piles[pile_index]

        # Validate the split
        if pile_size < 3 or x + y != pile_size or x == y or x <= 0 or y <= 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Apply the move
        self.piles[pile_index] = x
        # Find next empty spot for the new pile
        for i in range(self.MAX_N_PILES):
            if self.piles[i] == 0:
                self.piles[i] = y
                break
        else:
            # No space to add new pile
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check if the game is over
        if not any(self.piles[i] >= 3 for i in range(self.MAX_N_PILES)):
            # No valid moves left, current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Return a visual representation of the piles
        piles_list = [int(pile) for pile in self.piles if pile > 0]
        piles_str = "Current piles: " + str(piles_list)
        return piles_str
