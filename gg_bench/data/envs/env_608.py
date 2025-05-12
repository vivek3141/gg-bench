import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # 50 possible actions: 25 query actions (0-24), 25 guess actions (25-49)
        self.action_space = spaces.Discrete(50)

        # Observation space: 25 cells with values from -8 to 8
        # Positive values: hints for player 1
        # Negative values: hints for player -1 (player 2)
        # Zero: unqueried cell
        self.observation_space = spaces.Box(low=-8, high=8, shape=(25,), dtype=np.int8)

        self.grid_size = 5  # 5x5 grid
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.player_skip_turn = {1: False, -1: False}
        self.observation = np.zeros(25, dtype=np.int8)
        self.done = False
        # Randomly place the treasure
        self.treasure_location = self.np_random.integers(0, 25)
        return self.observation, {}

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Check if the current player should skip their turn
        if self.player_skip_turn[self.current_player]:
            self.player_skip_turn[self.current_player] = False
            # Switch to the next player
            self.current_player *= -1
            reward = 0  # No reward for skipped turn
            return self.observation, reward, False, False, {}

        # Validate action
        if action < 0 or action >= 50:
            # Invalid action
            return self.observation, -10, True, False, {}

        if action < 25:
            # Query action
            cell_index = action
            if self.observation[cell_index] != 0:
                # Cell has already been queried
                return self.observation, -10, True, False, {}
            # Calculate Manhattan distance to the treasure
            row_cell, col_cell = divmod(cell_index, self.grid_size)
            row_treasure, col_treasure = divmod(self.treasure_location, self.grid_size)
            distance = abs(row_cell - row_treasure) + abs(col_cell - col_treasure)
            # Update observation for the current player
            if self.current_player == 1:
                self.observation[cell_index] = distance
            else:
                self.observation[cell_index] = -distance
            reward = -10
            done = False
        else:
            # Guess action
            cell_index = action - 25
            if cell_index == self.treasure_location:
                # Correct guess
                reward = 1
                done = True
                self.done = True
            else:
                # Incorrect guess; player loses next turn
                self.player_skip_turn[self.current_player] = True
                reward = -10
                done = False
        # Switch to the next player
        self.current_player *= -1
        return self.observation, reward, done, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        grid_str = ""
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                cell_index = i * self.grid_size + j
                value = self.observation[cell_index]
                if value == 0:
                    cell_repr = " ? "
                else:
                    if value > 0:
                        cell_repr = f" {value} "
                    else:
                        cell_repr = f"-{-value} "
                row_str += cell_repr
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Return a list of valid moves as indices of the action_space
        moves = []
        # Add valid query actions (unqueried cells)
        for i in range(25):
            if self.observation[i] == 0:
                moves.append(i)  # Query actions (0-24)
        # Add all possible guess actions (allowed even if already guessed)
        for i in range(25):
            moves.append(i + 25)  # Guess actions (25-49)
        return moves
