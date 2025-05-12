import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The grid has 9 positions, numbered from 0 to 8
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=9, shape=(9,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid with numbers 1 to 9 randomly placed
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.grid_numbers = np.arange(1, 10)
        self.np_random.shuffle(self.grid_numbers)
        self.grid = self.grid_numbers.copy()

        # Initialize player scores
        self.player_scores = {1: 0, 2: 0}

        # Set the current player (Player 1 starts)
        self.current_player = 1

        # Game state flags
        self.done = False

        return self.get_observation(), {}

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {"message": "Game is over"}

        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self.get_observation(), -10, True, False, {"message": "Invalid move"}

        # Perform the move
        selected_number = self.grid[action]
        self.player_scores[self.current_player] += selected_number
        self.grid[action] = 0

        # Cascade removal of adjacent numbers
        self.cascade_removal(action, selected_number)

        # Check for win condition
        if self.player_scores[self.current_player] >= 15:
            self.done = True
            return (
                self.get_observation(),
                1,
                True,
                False,
                {"message": f"Player {self.current_player} wins!"},
            )

        # Check if all numbers are removed from the grid
        if np.all(self.grid == 0):
            self.done = True
            # Determine the winner based on scores and tiebreaker rules
            if self.player_scores[1] > self.player_scores[2]:
                winner = 1
            elif self.player_scores[2] > self.player_scores[1]:
                winner = 2
            else:
                # If scores are equal, the player who reached it first wins
                winner = self.current_player
            reward = 1 if winner == self.current_player else -10
            return (
                self.get_observation(),
                reward,
                True,
                False,
                {"message": f"Player {winner} wins!"},
            )

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        # Reward for a valid move (but not winning)
        reward = -10

        return self.get_observation(), reward, False, False, {}

    def get_adjacent_positions(self, index):
        row, col = divmod(index, 3)
        adjacent_positions = []

        # Up
        if row > 0:
            adjacent_positions.append(index - 3)
        # Down
        if row < 2:
            adjacent_positions.append(index + 3)
        # Left
        if col > 0:
            adjacent_positions.append(index - 1)
        # Right
        if col < 2:
            adjacent_positions.append(index + 1)

        return adjacent_positions

    def cascade_removal(self, index, selected_number):
        adjacent_positions = self.get_adjacent_positions(index)
        for pos in adjacent_positions:
            adj_number = self.grid[pos]
            if adj_number != 0 and adj_number < selected_number:
                self.grid[pos] = 0  # Remove the adjacent number

    def get_observation(self):
        return self.grid.copy()

    def valid_moves(self):
        return [i for i in range(9) if self.grid[i] != 0]

    def render(self):
        grid_representation = ""
        for i in range(3):
            row = ""
            for j in range(3):
                index = i * 3 + j
                num = self.grid[index]
                cell = f"[{num}]" if num != 0 else "[ ]"
                row += cell
            grid_representation += row + "\n"
        score_representation = (
            f"Player 1 Score: {self.player_scores[1]}\n"
            f"Player 2 Score: {self.player_scores[2]}\n"
            f"Current Player: {self.current_player}\n"
        )
        return grid_representation + score_representation
