import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=1, high=29, shape=(16,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly assign numbers 1-9 to each cell in the grid
        self.grid_numbers = np.random.randint(1, 10, size=(16,))

        # 0 = unclaimed, 1 = claimed by Player 1, -1 = claimed by Player 2
        self.claimed_cells = np.zeros(16, dtype=np.int32)

        # Player sums
        self.player_sums = {1: 0, -1: 0}

        # Current player: 1 for Player 1, -1 for Player 2
        self.current_player = 1

        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        if not self._is_valid_action(action):
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Claim the cell
        self.claimed_cells[action] = self.current_player
        self.player_sums[self.current_player] += self.grid_numbers[action]

        # Check for victory or defeat
        reward = 0
        if self.player_sums[self.current_player] == 15:
            reward = 1  # Current player wins
            self.done = True
            return self._get_observation(), reward, True, False, {}
        elif self.player_sums[self.current_player] > 15:
            reward = -1  # Current player loses
            self.done = True
            return self._get_observation(), reward, True, False, {}

        elif np.all(self.claimed_cells != 0):
            # All cells are claimed, determine winner
            self.done = True
            player1_sum = self.player_sums[1] if self.player_sums[1] <= 15 else 0
            player2_sum = self.player_sums[-1] if self.player_sums[-1] <= 15 else 0
            if player1_sum > player2_sum:
                reward = 1 if self.current_player == 1 else -1
            elif player2_sum > player1_sum:
                reward = 1 if self.current_player == -1 else -1
            else:
                reward = 0  # Draw
            return self._get_observation(), reward, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return self._get_observation(), 0, False, False, {}

    def render(self):
        grid_display = ""
        for row in range(4):
            grid_display += "+---+---+---+---+\n"
            grid_display += "|"
            for col in range(4):
                idx = row * 4 + col
                num = self.grid_numbers[idx]
                claim = self.claimed_cells[idx]
                if claim == 1:
                    cell_display = f"X{num}"
                elif claim == -1:
                    cell_display = f"O{num}"
                else:
                    cell_display = f" {num}"
                grid_display += f"{cell_display:^3}|"
            grid_display += "\n"
        grid_display += "+---+---+---+---+\n"
        grid_display += f"Player 1 sum: {self.player_sums[1]}, Player 2 sum: {self.player_sums[-1]}\n"
        print(grid_display)

    def valid_moves(self):
        valid_actions = []
        for action in range(16):
            if self._is_valid_action(action):
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        obs = self.grid_numbers.copy()
        for i in range(16):
            if self.claimed_cells[i] == 1:
                obs[i] += 10  # Values 11-19
            elif self.claimed_cells[i] == -1:
                obs[i] += 20  # Values 21-29
        return obs

    def _is_valid_action(self, action):
        if self.claimed_cells[action] != 0:
            return False  # Cell already claimed

        if np.sum(self.claimed_cells == self.current_player) == 0:
            # First move, any unclaimed cell is valid
            return True

        # Check adjacency to claimed cells
        claimed_indices = np.where(self.claimed_cells == self.current_player)[0]
        for idx in claimed_indices:
            if self._are_adjacent(idx, action):
                return True
        return False

    def _are_adjacent(self, idx1, idx2):
        row1, col1 = divmod(idx1, 4)
        row2, col2 = divmod(idx2, 4)
        return (abs(row1 - row2) + abs(col1 - col2)) == 1
