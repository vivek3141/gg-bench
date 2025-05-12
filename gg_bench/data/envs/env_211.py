import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-Up, 1-Down, 2-Left, 3-Right
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Grid is 5x5, each cell can be 0 (empty), 1 (star), 2 (Player 1), 3 (Player 2)
        # Additional 3 entries for Player 1 stars, Player 2 stars, Current player
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(25 + 3,), dtype=np.int8
        )

        self.grid_size = 5
        self.num_stars = 7

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Place stars randomly, excluding starting positions
        available_positions = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in [(0, 0), (4, 4)]
        ]
        self.rng, _ = gym.utils.seeding.np_random(seed)
        star_positions = self.rng.choice(
            len(available_positions), self.num_stars, replace=False
        )
        for idx in star_positions:
            pos = available_positions[idx]
            self.grid[pos] = 1  # Place a star

        # Set player positions
        self.player_positions = {
            1: (0, 0),  # Player 1 starting position
            2: (4, 4),  # Player 2 starting position
        }

        self.grid[self.player_positions[1]] = 2  # Mark Player 1 on the grid
        self.grid[self.player_positions[2]] = 3  # Mark Player 2 on the grid

        # Reset star counts
        self.player_star_counts = {
            1: 0,
            2: 0,
        }

        # Set current player
        self.current_player = 1

        # Game status
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to movement
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if action not in [0, 1, 2, 3]:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        move = move_map[action]
        curr_pos = self.player_positions[self.current_player]
        new_pos = (curr_pos[0] + move[0], curr_pos[1] + move[1])

        # Check if new position is within bounds
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Remove player from current position on the grid
        self.grid[curr_pos] = 0

        # Check for interactions at new position
        cell_value = self.grid[new_pos]

        # Initialize reward
        reward = 0

        # If cell has a star
        if cell_value == 1:
            self.player_star_counts[self.current_player] += 1
            # Remove star from the grid
            self.grid[new_pos] = 0

        # If cell has the opponent
        elif cell_value in [2, 3]:
            opponent = 1 if self.current_player == 2 else 2
            # Steal a star if opponent has any
            if self.player_star_counts[opponent] > 0:
                self.player_star_counts[self.current_player] += 1
                self.player_star_counts[opponent] -= 1

        # Update player's position
        self.player_positions[self.current_player] = new_pos
        # Mark player on the grid
        self.grid[new_pos] = 2 if self.current_player == 1 else 3

        # Check for win condition
        if self.player_star_counts[self.current_player] >= 3:
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = 1 if self.current_player == 2 else 2

        return self._get_observation(), reward, False, False, {}

    def render(self):
        symbols = {
            0: " . ",  # Empty cell
            1: " * ",  # Star
            2: "P1 ",  # Player 1
            3: "P2 ",  # Player 2
        }
        render_str = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                render_str += symbols[self.grid[i, j]]
            render_str += "\n"
        render_str += f"Player 1 Stars: {self.player_star_counts[1]}\n"
        render_str += f"Player 2 Stars: {self.player_star_counts[2]}\n"
        render_str += f"Current Player: Player {self.current_player}\n"
        return render_str

    def valid_moves(self):
        valid_actions = []
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        curr_pos = self.player_positions[self.current_player]
        for action, move in move_map.items():
            new_pos = (curr_pos[0] + move[0], curr_pos[1] + move[1])
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Flatten the grid and append player information
        grid_flat = self.grid.flatten()
        player_info = np.array(
            [
                self.player_star_counts[1],
                self.player_star_counts[2],
                self.current_player,
            ],
            dtype=np.int8,
        )
        observation = np.concatenate((grid_flat, player_info))
        return observation
