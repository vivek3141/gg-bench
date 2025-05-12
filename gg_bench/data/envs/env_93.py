import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define grid size
        self.grid_size = 5  # 5x5 grid

        # Define action and observation space
        # There are 50 actions: 25 SCAN actions and 25 GUESS actions
        self.action_space = spaces.Discrete(50)

        # Observation space: For each position in the grid
        # -1 indicates unscanned position
        # 0-8 indicates the distance after a scan
        self.observation_space = spaces.Box(low=-1, high=8, shape=(25,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly place the treasure
        self.treasure_position = self.np_random.integers(0, 25)

        # Initialize observations for both players
        self.player_observations = [
            np.full(25, -1, dtype=np.int8),
            np.full(25, -1, dtype=np.int8),
        ]

        # Set the current player (0 or 1)
        self.current_player = 0

        # Game state
        self.done = False

        # Return the initial observation and info
        return self.player_observations[self.current_player], {}

    def step(self, action):
        if self.done:
            # Game is over
            observation = self.player_observations[self.current_player]
            return observation, 0, True, False, {}

        # Map action to action type and position
        if 0 <= action < 25:
            action_type = "SCAN"
            position = action
        elif 25 <= action < 50:
            action_type = "GUESS"
            position = action - 25
        else:
            # Invalid action
            reward = -10
            terminated = True
            observation = self.player_observations[self.current_player]
            return observation, reward, terminated, False, {}

        if action_type == "SCAN":
            # Compute the Manhattan distance
            treasure_row, treasure_col = self.position_to_coords(self.treasure_position)
            position_row, position_col = self.position_to_coords(position)

            distance = abs(treasure_row - position_row) + abs(
                treasure_col - position_col
            )

            # Update the player's observation
            self.player_observations[self.current_player][position] = distance

            reward = 0  # No reward for a valid scan
            terminated = False
            truncated = False

        elif action_type == "GUESS":
            if position == self.treasure_position:
                # Correct guess, player wins
                reward = 1
                terminated = True
                truncated = False
            else:
                # Incorrect guess, player loses
                reward = -10
                terminated = True
                truncated = False

        # Prepare the observation
        observation = self.player_observations[self.current_player]

        # If the game is not over, switch to the next player
        if not terminated:
            self.current_player = 1 - self.current_player

        self.done = terminated

        return observation, reward, terminated, False, {}

    def render(self):
        # Generate the grid representation for the current player
        grid_display = ""
        observation = self.player_observations[self.current_player]

        grid_display += "  1  2  3  4  5\n"
        for row in range(self.grid_size):
            grid_display += chr(ord("A") + row) + " "
            for col in range(self.grid_size):
                position = self.coords_to_position(row, col)
                value = observation[position]
                if value == -1:
                    grid_display += " - "
                else:
                    grid_display += f" {value} "
            grid_display += "\n"
        return grid_display

    def valid_moves(self):
        # All actions are valid unless the game is over
        if self.done:
            return []
        else:
            return list(range(50))

    def position_to_coords(self, position):
        row = position // self.grid_size
        col = position % self.grid_size
        return row, col

    def coords_to_position(self, row, col):
        return row * self.grid_size + col
