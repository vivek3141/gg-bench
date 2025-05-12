import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0: Move up, 1: Move down, 2: Move left,
        # 3: Move right, 4: Attack
        self.action_space = spaces.Discrete(5)

        # Define observation space:
        # [P1_x, P1_y, P1_shields, P2_x, P2_y, P2_shields]
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 0, 1, 1, 0]),
            high=np.array([5, 5, 3, 5, 5, 3]),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Player positions
        self.player_positions = {
            1: [1, 1],  # Player 1 starts at (1,1)
            2: [5, 5],  # Player 2 starts at (5,5)
        }
        # Player shields
        self.player_shields = {1: 3, 2: 3}
        # Current player: 1 or 2
        self.current_player = 1
        # Game over flag
        self.done = False
        # Build the observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            observation = self._get_observation()
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move, game over
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, {}

        opponent = 2 if self.current_player == 1 else 1
        reward = -10  # Default reward for a valid move

        if action == 4:  # Attack
            # Reduce opponent's shields by one
            self.player_shields[opponent] -= 1
        else:
            # Move action
            moved = self._move_player(action)
            # Check for capturing opponent
            if (
                self.player_positions[self.current_player]
                == self.player_positions[opponent]
            ):
                if self.player_shields[opponent] == 0:
                    # Current player wins
                    self.done = True
                    reward = 1  # Reward for winning
                else:
                    # Cannot move onto opponent with shields > 0
                    self.done = True
                    observation = self._get_observation()
                    return observation, -10, True, False, {}

        # Switch to the other player
        self.current_player = opponent
        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        grid = [["   " for _ in range(5)] for _ in range(5)]
        p1_x, p1_y = self.player_positions[1]
        p2_x, p2_y = self.player_positions[2]

        # Place Player 1
        grid[5 - p1_y][p1_x - 1] = "P1 "
        # Place Player 2
        grid[5 - p2_y][p2_x - 1] = "P2 "

        grid_str = "    1   2   3   4   5\n"
        grid_str += "  -------------------------\n"
        for idx, row in enumerate(grid):
            grid_str += f"{5 - idx} |" + "|".join(row) + "|\n"
            grid_str += "  -------------------------\n"
        grid_str += (
            f"Player 1 Position: ({p1_x},{p1_y}), Shields: {self.player_shields[1]}\n"
        )
        grid_str += (
            f"Player 2 Position: ({p2_x},{p2_y}), Shields: {self.player_shields[2]}\n"
        )
        grid_str += f"Current Player: Player {self.current_player}\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        opponent = 2 if self.current_player == 1 else 1
        cp_x, cp_y = self.player_positions[self.current_player]
        op_x, op_y = self.player_positions[opponent]
        op_shields = self.player_shields[opponent]

        # Directions: 0: up, 1: down, 2: left, 3: right
        directions = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}

        # Check movement actions
        for action in range(4):
            dx, dy = directions[action]
            new_x = cp_x + dx
            new_y = cp_y + dy

            # Grid wrapping
            if new_x < 1:
                new_x = 5
            elif new_x > 5:
                new_x = 1
            if new_y < 1:
                new_y = 5
            elif new_y > 5:
                new_y = 1

            # Check if moving onto opponent
            if [new_x, new_y] == [op_x, op_y]:
                if op_shields == 0:
                    # Can capture opponent
                    valid_actions.append(action)
            else:
                valid_actions.append(action)

        # Check if attack is valid
        if self.player_shields[opponent] > 0:
            valid_actions.append(4)  # Attack action

        return valid_actions

    def _get_observation(self):
        p1_x, p1_y = self.player_positions[1]
        p2_x, p2_y = self.player_positions[2]
        p1_shields = self.player_shields[1]
        p2_shields = self.player_shields[2]
        observation = np.array(
            [p1_x, p1_y, p1_shields, p2_x, p2_y, p2_shields], dtype=np.int32
        )
        return observation

    def _move_player(self, action):
        # Move the current player
        cp = self.current_player
        opponent = 2 if cp == 1 else 1
        cp_x, cp_y = self.player_positions[cp]
        op_x, op_y = self.player_positions[opponent]
        directions = {
            0: (0, 1),  # Up
            1: (0, -1),  # Down
            2: (-1, 0),  # Left
            3: (1, 0),  # Right
        }
        dx, dy = directions[action]
        new_x = cp_x + dx
        new_y = cp_y + dy

        # Grid wrapping
        if new_x < 1:
            new_x = 5
        elif new_x > 5:
            new_x = 1
        if new_y < 1:
            new_y = 5
        elif new_y > 5:
            new_y = 1

        # Check if moving onto opponent with shields > 0
        if [new_x, new_y] == [op_x, op_y] and self.player_shields[opponent] > 0:
            return False  # Invalid move
        else:
            self.player_positions[cp] = [new_x, new_y]
            return True
