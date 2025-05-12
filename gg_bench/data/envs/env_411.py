import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 6 possible actions: claim one of the 3 rows or 3 columns
        self.action_space = spaces.Discrete(6)

        # Observation space consists of the grid numbers (1-9) and the claimed lines status (-1, 0, 1)
        # First 9 elements are the grid numbers, next 6 elements are the claimed lines
        self.observation_space = spaces.Box(
            low=np.array([1] * 9 + [-1] * 6),
            high=np.array([9] * 9 + [1] * 6),
            shape=(15,),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Generate a random arrangement of numbers 1-9
        self.grid_numbers = rng.permutation(np.arange(1, 10))
        self.grid_numbers = self.grid_numbers.reshape((3, 3))

        # Initialize claimed lines: 0 = unclaimed, 1 = claimed by player 1, -1 = claimed by player 2
        self.claimed_lines = np.zeros(6, dtype=np.int32)

        # Keep track of lines claimed by each player
        self.lines_claimed_by_player = {1: 0, -1: 0}

        # Game over flag
        self.done = False

        # Current player (1 or -1)
        self.current_player = 1

        # Prepare the observation
        self._update_observation()

        return self.observation, {}

    def step(self, action):
        # Check if the action is valid
        if self.done or action not in self.valid_moves():
            self.done = True
            reward = -10  # Invalid move
            return self.observation, reward, True, False, {}

        # Claim the line for the current player
        self.claimed_lines[action] = self.current_player
        self.lines_claimed_by_player[self.current_player] += 1

        # Check if the game is over (both players have claimed 3 lines)
        if (self.lines_claimed_by_player[1] + self.lines_claimed_by_player[-1]) >= 6:
            self.done = True
            # Calculate scores
            player1_score = self._calculate_score(1)
            player2_score = self._calculate_score(-1)

            if player1_score > player2_score:
                winner = 1
            elif player2_score > player1_score:
                winner = -1
            else:
                # Tie-breaker: Player who moved second wins
                winner = -self.current_player  # Since current_player made the last move
            if winner == self.current_player:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses or tie
            self._update_observation()
            return self.observation, reward, True, False, {}
        else:
            # Switch the current player
            self.current_player *= -1
            reward = 0  # No reward for a regular valid move
            self._update_observation()
            return self.observation, reward, False, False, {}

    def render(self):
        # Build the grid display
        grid_display = "+-----+-----+-----+\n"
        for i in range(3):
            grid_display += "|"
            for j in range(3):
                grid_display += f"  {self.grid_numbers[i][j]}  |"
            grid_display += f"  Row {i+1}\n+-----+-----+-----+\n"
        grid_display += "  C1    C2    C3\n"

        # Display claimed lines
        lines = ["Row1", "Row2", "Row3", "Col1", "Col2", "Col3"]
        claimed_status = {1: "Player 1", -1: "Player 2", 0: "Unclaimed"}
        claimed_lines_display = "Claimed Lines:\n"
        for idx, status in enumerate(self.claimed_lines):
            claimed_lines_display += f"{lines[idx]}: {claimed_status[status]}\n"

        # Display current player
        current_player_display = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"

        return grid_display + claimed_lines_display + current_player_display

    def valid_moves(self):
        # Valid actions are the indices of unclaimed lines
        return [idx for idx, status in enumerate(self.claimed_lines) if status == 0]

    def _calculate_score(self, player):
        score = 0
        lines = [
            self.grid_numbers[0, :],  # Row 1
            self.grid_numbers[1, :],  # Row 2
            self.grid_numbers[2, :],  # Row 3
            self.grid_numbers[:, 0],  # Column 1
            self.grid_numbers[:, 1],  # Column 2
            self.grid_numbers[:, 2],  # Column 3
        ]
        for idx, status in enumerate(self.claimed_lines):
            if status == player:
                line = lines[idx]
                score += np.sum(line)
        return score

    def _update_observation(self):
        # Prepare the observation array
        grid_flat = self.grid_numbers.flatten()
        observation = np.concatenate((grid_flat, self.claimed_lines))
        self.observation = observation.astype(np.int32)
