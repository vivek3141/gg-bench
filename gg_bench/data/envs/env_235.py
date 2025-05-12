import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete space of 21 actions (move forward by 1 to 21 cells)
        self.action_space = spaces.Discrete(21)

        # Observation space: Positions of both players (from 0 to 21, inclusive)
        # Positions are represented as an array: [position_player1, position_player2]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([21, 21]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Both players start off the grid at position 0
        self.positions = np.array([0, 0], dtype=np.int32)
        self.current_player = 0  # Player 1 starts (index 0)
        self.done = False
        return self.positions.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.positions.copy(), 0, True, False, {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        player = self.current_player
        other_player = 1 - player  # Switch between 0 and 1

        move_forward = (
            action + 1
        )  # Actions are 0-indexed (action 0 means move forward by 1)

        current_pos = self.positions[player]
        new_pos = current_pos + move_forward
        self.positions[player] = new_pos

        if new_pos == 21:
            # Current player wins
            self.done = True
            return self.positions.copy(), 1, True, False, {}
        else:
            # Switch to the other player
            self.current_player = other_player
            return self.positions.copy(), 0, False, False, {}

    def get_divisors(self, n):
        # Return a list of divisors of n
        return [i for i in range(1, n + 1) if n % i == 0]

    def valid_moves(self):
        if self.done:
            return []

        player = self.current_player
        current_pos = self.positions[player]

        if current_pos == 0:
            # Player is off the grid; the only valid action is to enter at cell 1
            return [0]  # Action 0 corresponds to moving forward by 1
        else:
            current_cell = current_pos
            divisors = self.get_divisors(current_cell)
            valid_moves = []

            for move in divisors:
                new_pos = current_cell + move
                if new_pos <= 21:
                    valid_moves.append(move)

            # Map move_forward to action indices
            valid_actions = [move - 1 for move in valid_moves]
            return valid_actions

    def render(self):
        grid = []
        for i in range(1, 22):  # Cells 1 to 21
            players_here = []
            if self.positions[0] == i:
                players_here.append("X")
            if self.positions[1] == i:
                players_here.append("O")
            if players_here:
                cell_content = ",".join(players_here)
                cell_str = f"[{cell_content}]"
            else:
                cell_str = "[ ]"
            grid.append(cell_str)
        grid_str = "".join(grid)
        return (
            f"Current Player: {'X' if self.current_player == 0 else 'O'}\n" + grid_str
        )
