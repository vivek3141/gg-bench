import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Actions: 8 possible movements
        self.action_space = spaces.Discrete(8)

        # Observation space: [Seeker_row, Seeker_col, current_player (0 or 1), own_goal_row, own_goal_col]
        self.observation_space = spaces.Box(low=0, high=4, shape=(5,), dtype=np.int32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Seeker starts at position (2,2)
        self.seeker_pos = [2, 2]

        # Randomly assign Goal Cells
        self.player1_goal_col = self.np_random.integers(0, 5)
        self.player1_goal_cell = [4, self.player1_goal_col]

        self.player2_goal_col = self.np_random.integers(0, 5)
        self.player2_goal_cell = [0, self.player2_goal_col]

        self.goal_cells = {1: self.player1_goal_cell, 2: self.player2_goal_cell}

        self.current_player = 1

        self.done = False

        own_goal_cell = self.goal_cells[self.current_player]

        observation = np.array(
            [
                self.seeker_pos[0],
                self.seeker_pos[1],
                self.current_player - 1,
                own_goal_cell[0],
                own_goal_cell[1],
            ],
            dtype=np.int32,
        )

        return observation, {}

    def step(self, action):
        # Define the possible moves
        moves = [
            (-1, 0),  # Up
            (-1, +1),  # Up-Right
            (0, +1),  # Right
            (+1, +1),  # Down-Right
            (+1, 0),  # Down
            (+1, -1),  # Down-Left
            (0, -1),  # Left
            (-1, -1),  # Up-Left
        ]

        info = {}

        if self.done:
            # The game has already ended
            own_goal_cell = self.goal_cells[self.current_player]
            observation = np.array(
                [
                    self.seeker_pos[0],
                    self.seeker_pos[1],
                    self.current_player - 1,
                    own_goal_cell[0],
                    own_goal_cell[1],
                ],
                dtype=np.int32,
            )
            return observation, 0, True, False, info

        move = moves[action]
        new_row = self.seeker_pos[0] + move[0]
        new_col = self.seeker_pos[1] + move[1]

        if new_row < 0 or new_row > 4 or new_col < 0 or new_col > 4:
            # Invalid move
            reward = -10
            self.done = True
            own_goal_cell = self.goal_cells[self.current_player]
            observation = np.array(
                [
                    self.seeker_pos[0],
                    self.seeker_pos[1],
                    self.current_player - 1,
                    own_goal_cell[0],
                    own_goal_cell[1],
                ],
                dtype=np.int32,
            )
            return observation, reward, True, False, info
        else:
            # Valid move
            self.seeker_pos = [new_row, new_col]

            if self.seeker_pos == self.goal_cells[self.current_player]:
                # Current player wins
                reward = 1
                self.done = True

                own_goal_cell = self.goal_cells[self.current_player]
                observation = np.array(
                    [
                        self.seeker_pos[0],
                        self.seeker_pos[1],
                        self.current_player - 1,
                        own_goal_cell[0],
                        own_goal_cell[1],
                    ],
                    dtype=np.int32,
                )
                return observation, reward, True, False, info
            else:
                # Switch player
                self.current_player = 2 if self.current_player == 1 else 1
                reward = 0
                own_goal_cell = self.goal_cells[self.current_player]
                observation = np.array(
                    [
                        self.seeker_pos[0],
                        self.seeker_pos[1],
                        self.current_player - 1,
                        own_goal_cell[0],
                        own_goal_cell[1],
                    ],
                    dtype=np.int32,
                )
                return observation, reward, False, False, info

    def render(self):
        grid = [["   " for _ in range(5)] for _ in range(5)]
        row, col = self.seeker_pos
        grid[row][col] = " S "
        grid_str = "  +---+---+---+---+---+\n"
        for row_cells in grid:
            grid_str += "  |" + "|".join(row_cells) + "|\n"
            grid_str += "  +---+---+---+---+---+\n"
        return grid_str

    def valid_moves(self):
        moves = [
            (-1, 0),  # Up
            (-1, +1),  # Up-Right
            (0, +1),  # Right
            (+1, +1),  # Down-Right
            (+1, 0),  # Down
            (+1, -1),  # Down-Left
            (0, -1),  # Left
            (-1, -1),  # Up-Left
        ]

        valid_actions = []
        for idx, move in enumerate(moves):
            new_row = self.seeker_pos[0] + move[0]
            new_col = self.seeker_pos[1] + move[1]
            if 0 <= new_row <= 4 and 0 <= new_col <= 4:
                valid_actions.append(idx)
        return valid_actions
