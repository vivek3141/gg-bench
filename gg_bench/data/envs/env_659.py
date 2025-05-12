import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 9 possible actions for 9 cells
        self.action_space = spaces.Discrete(9)

        # Define observation space:
        # - First 9 elements: grid cells (revealed numbers or 0 if hidden)
        # - 10th element: current player's total score
        # - 11th element: opponent's total score
        self.observation_space = spaces.Box(low=0, high=20, shape=(11,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign numbers 1 to 9 to the grid
        numbers = np.arange(1, 10)
        self.np_random.shuffle(numbers)
        self.grid_numbers = numbers  # Hidden numbers

        # Track selected cells and revealed numbers
        self.cell_selected = np.zeros(9, dtype=bool)
        self.grid_revealed = np.zeros(9, dtype=np.int32)  # 0 if not revealed

        # Initialize scores
        self.player1_total = 0
        self.player2_total = 0

        # Current player's turn (1 or 2)
        self.current_player = 1

        self.done = False

        return self._get_observation(), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Validate action
        if not self.action_space.contains(action) or self.cell_selected[action]:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Reveal the number in the selected cell
        number = self.grid_numbers[action]
        self.cell_selected[action] = True
        self.grid_revealed[action] = number

        # Update the player's score
        if self.current_player == 1:
            self.player1_total += number
            player_total = self.player1_total
        else:
            self.player2_total += number
            player_total = self.player2_total

        # Check for winning or losing conditions
        if player_total == 15:
            # Current player wins
            self.done = True
            reward = 1
        elif player_total > 15:
            # Current player loses
            self.done = True
            reward = -1
        else:
            reward = 0

        # Check if all cells have been selected
        if not self.done and np.all(self.cell_selected):
            self.done = True
            # Determine winner based on closest to 15 without exceeding
            p1_diff = (
                15 - self.player1_total if self.player1_total <= 15 else float("inf")
            )
            p2_diff = (
                15 - self.player2_total if self.player2_total <= 15 else float("inf")
            )
            if p1_diff < p2_diff:
                winner = 1
            elif p2_diff < p1_diff:
                winner = 2
            else:
                # Tie-breaker: last player to play loses
                winner = 2 if self.current_player == 1 else 1
            reward = 1 if winner == self.current_player else -1

        # Switch player if game is not over
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        observation = self._get_observation()
        terminated = self.done
        truncated = False

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        # Construct the observation array
        obs = np.zeros(11, dtype=np.int32)
        obs[:9] = self.grid_revealed
        if self.current_player == 1:
            obs[9] = self.player1_total
            obs[10] = self.player2_total
        else:
            obs[9] = self.player2_total
            obs[10] = self.player1_total
        return obs

    def render(self):
        # Generate a visual representation of the game state
        grid_display = ""
        for i in range(3):
            row = ""
            for j in range(3):
                idx = i * 3 + j
                if self.cell_selected[idx]:
                    row += f" {self.grid_revealed[idx]} "
                else:
                    row += " [ ] "
            grid_display += row + "\n"
        scores = (
            f"Player 1 Total Score: {self.player1_total}\n"
            f"Player 2 Total Score: {self.player2_total}\n"
            f"Current Player: Player {self.current_player}\n"
        )
        return grid_display + scores

    def valid_moves(self):
        # Return a list of valid moves (unselected cell indices)
        return [i for i in range(9) if not self.cell_selected[i]]
