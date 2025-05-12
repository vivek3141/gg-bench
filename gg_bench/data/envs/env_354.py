import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            81
        )  # 72 two-number actions + 9 one-number actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(18,), dtype=np.int8)

        # Initialize the action mappings
        self.numbers = list(range(1, 10))  # Numbers 1 to 9
        self.action_to_tuple = {}  # Maps action index to (num1, num2, operation)
        self.tuple_to_action = {}  # Maps (num1, num2, operation) to action index
        action_index = 0

        # Generate all combinations of two numbers and operations
        pairs = list(itertools.combinations(self.numbers, 2))  # 36 pairs
        for pair in pairs:
            for operation in ["add", "subtract"]:
                self.action_to_tuple[action_index] = (pair[0], pair[1], operation)
                self.tuple_to_action[(pair[0], pair[1], operation)] = action_index
                action_index += 1

        # Generate actions for single numbers (when only one number remains)
        for num in self.numbers:
            self.action_to_tuple[action_index] = (num, None, None)
            self.tuple_to_action[(num, None, None)] = action_index
            action_index += 1  # action_index from 72 to 80

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize grids with numbers 1-9 randomly shuffled
        self.player_grids = {}
        self.available_numbers = self.numbers.copy()

        grid_numbers = self.numbers.copy()
        np.random.shuffle(grid_numbers)
        self.player_grids[1] = grid_numbers.copy()

        grid_numbers = self.numbers.copy()
        np.random.shuffle(grid_numbers)
        self.player_grids[-1] = grid_numbers.copy()

        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Get action details
        if action not in self.action_to_tuple:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        num1, num2, operation = self.action_to_tuple[action]

        player_grid = self.player_grids[self.current_player]
        opponent_grid = self.player_grids[-self.current_player]

        # Check if selected numbers are in player's grid
        if num2 is None:
            # Single number action
            if num1 not in player_grid or len(player_grid) != 1:
                self.done = True
                return self._get_observation(), -10, True, False, {}
            result = num1
            used_numbers = [num1]
        else:
            # Two-number action
            if num1 not in player_grid or num2 not in player_grid or num1 == num2:
                self.done = True
                return self._get_observation(), -10, True, False, {}

            if operation == "add":
                result = num1 + num2
            else:
                result = abs(num1 - num2)

            used_numbers = [num1, num2]

        # Remove used numbers from current player's grid
        for num in used_numbers:
            player_grid.remove(num)

        # Check if result is in opponent's grid
        if result in opponent_grid:
            opponent_grid.remove(result)

        # Check for win condition
        if not opponent_grid:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch player and continue
            self.current_player *= -1
            observation = self._get_observation()
            return observation, 0, False, False, {}

    def render(self):
        player_grid = self.player_grids[self.current_player]
        opponent_grid = self.player_grids[-self.current_player]
        render_str = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        render_str += f"Your Grid: {sorted(player_grid)}\n"
        render_str += f"Opponent's Grid: {sorted(opponent_grid)}\n"
        return render_str

    def valid_moves(self):
        player_grid = self.player_grids[self.current_player]
        valid_action_indices = []

        if len(player_grid) >= 2:
            # Generate all combinations of two numbers from player's grid
            pairs = list(itertools.combinations(sorted(player_grid), 2))
            for pair in pairs:
                for operation in ["add", "subtract"]:
                    action = self.tuple_to_action.get((pair[0], pair[1], operation))
                    if action is not None:
                        valid_action_indices.append(action)
        elif len(player_grid) == 1:
            # Only one number remains
            num = player_grid[0]
            action = self.tuple_to_action.get((num, None, None))
            if action is not None:
                valid_action_indices.append(action)
        return valid_action_indices

    def _get_observation(self):
        # Observation is a 18-length array
        # First 9 positions represent current player's grid numbers (1 if present, 0 if not)
        # Next 9 positions represent opponent's grid numbers
        observation = np.zeros(18, dtype=np.int8)
        player_grid = self.player_grids[self.current_player]
        opponent_grid = self.player_grids[-self.current_player]

        for num in player_grid:
            observation[num - 1] = 1  # Indices 0-8 for numbers 1-9

        for num in opponent_grid:
            observation[9 + num - 1] = 1  # Indices 9-17 for opponent's numbers

        return observation
