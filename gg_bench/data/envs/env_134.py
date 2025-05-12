import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are the 8 possible movements
        self.action_space = spaces.Discrete(8)
        # Observation: [current_row, current_col, opponent_row, opponent_col]
        self.observation_space = spaces.Box(low=1, high=5, shape=(4,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a hidden path from (1,1) to (5,5)
        self.hidden_path = self._generate_hidden_path()

        # Initialize player positions and progress
        self.player_positions = {1: (1, 1), -1: (1, 1)}  # Player 1 and Player -1
        self.player_progress = {1: 0, -1: 0}  # Index in the hidden path
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            observation = self._get_observation()
            reward = 0
            info = {}
            return observation, reward, True, False, info

        # Get current position
        current_pos = self.player_positions[self.current_player]
        move = self._action_to_move(action)
        new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

        # Check if move is within grid boundaries
        if not (1 <= new_pos[0] <= 5 and 1 <= new_pos[1] <= 5):
            reward = -10
            info = {
                "feedback": "Invalid move. You can only move within the grid. Turn forfeited."
            }
            self._switch_player()
            observation = self._get_observation()
            return observation, reward, False, False, info

        # Check if move is adjacent
        if abs(move[0]) > 1 or abs(move[1]) > 1:
            reward = -10
            info = {
                "feedback": "Invalid move. You can only move to an adjacent cell. Turn forfeited."
            }
            self._switch_player()
            observation = self._get_observation()
            return observation, reward, False, False, info

        # Check if moving into opponent's cell (except Start Point)
        opponent_pos = self.player_positions[-self.current_player]
        if new_pos == opponent_pos and new_pos != (1, 1):
            reward = -10
            info = {
                "feedback": "Invalid move. Cell occupied by opponent. Turn forfeited."
            }
            self._switch_player()
            observation = self._get_observation()
            return observation, reward, False, False, info

        # Check if move is the correct next cell in the hidden path
        player_progress = self.player_progress[self.current_player]
        next_correct_pos = self.hidden_path[player_progress + 1]

        if new_pos == next_correct_pos:
            # Correct move
            self.player_positions[self.current_player] = new_pos
            self.player_progress[self.current_player] += 1
            info = {"feedback": "Correct"}

            if new_pos == (5, 5):
                # Player wins
                reward = 1
                self.done = True
                observation = self._get_observation()
                return observation, reward, True, False, info
            else:
                reward = 0
        else:
            # Incorrect move, return to previous position
            info = {"feedback": "Incorrect move. Return to previous position."}
            reward = 0

        self._switch_player()
        observation = self._get_observation()
        return observation, reward, self.done, False, info

    def render(self):
        grid = [["   " for _ in range(5)] for _ in range(5)]
        for player, pos in self.player_positions.items():
            symbol = "P1" if player == 1 else "P2"
            row, col = pos
            grid[row - 1][col - 1] = f" {symbol}"
        grid_str = "Grid:\n"
        for row in grid:
            grid_str += " |".join(row) + "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        current_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[-self.current_player]
        for action in range(8):
            move = self._action_to_move(action)
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            # Check grid boundaries
            if not (1 <= new_pos[0] <= 5 and 1 <= new_pos[1] <= 5):
                continue
            # Check if move is adjacent
            if abs(move[0]) > 1 or abs(move[1]) > 1:
                continue
            # Check if moving into opponent's cell (except Start Point)
            if new_pos == opponent_pos and new_pos != (1, 1):
                continue
            valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        current_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[-self.current_player]
        observation = np.array(
            [current_pos[0], current_pos[1], opponent_pos[0], opponent_pos[1]],
            dtype=np.int32,
        )
        return observation

    def _action_to_move(self, action):
        action_map = {
            0: (-1, -1),  # Up-Left
            1: (-1, 0),  # Up
            2: (-1, 1),  # Up-Right
            3: (0, -1),  # Left
            4: (0, 1),  # Right
            5: (1, -1),  # Down-Left
            6: (1, 0),  # Down
            7: (1, 1),  # Down-Right
        }
        return action_map[action]

    def _switch_player(self):
        self.current_player *= -1

    def _generate_hidden_path(self):
        from random import choice, seed

        seed()

        def get_neighbors(cell, visited):
            neighbors = []
            row, col = cell
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 1 <= new_row <= 5 and 1 <= new_col <= 5:
                        if (new_row, new_col) not in visited:
                            neighbors.append((new_row, new_col))
            return neighbors

        path = [(1, 1)]
        visited = set(path)
        while path[-1] != (5, 5):
            current_cell = path[-1]
            neighbors = get_neighbors(current_cell, visited)
            if not neighbors:
                # Backtrack
                visited.remove(path.pop())
                if not path:
                    # Start over if backtracked to the beginning
                    return self._generate_hidden_path()
            else:
                next_cell = choice(neighbors)
                path.append(next_cell)
                visited.add(next_cell)
        return path
