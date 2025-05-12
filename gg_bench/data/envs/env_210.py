import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0-9 correspond to selecting numbers 1-10
        self.action_space = spaces.Discrete(10)
        # Observation space represents the total sums of both players
        self.observation_space = spaces.Box(low=0, high=31, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_totals = [0, 0]  # Player 1 and Player 2 totals
        self.current_player = 0  # 0 for Player 1's turn, 1 for Player 2
        self.done = False
        return np.array(self.player_totals, dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(self.player_totals, dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            reward = -10
            return (
                np.array(self.player_totals, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        selected_number = action + 1  # Map action (0-9) to number (1-10)

        # Update total sum for the current player
        new_total = self.player_totals[self.current_player] + selected_number

        # Check for exceeding 31
        if new_total > 31:
            # Current player loses
            self.player_totals[self.current_player] = new_total
            self.done = True
            reward = -10
            return (
                np.array(self.player_totals, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        # Check for winning condition
        elif new_total in [23, 29, 31]:
            # Current player wins
            self.player_totals[self.current_player] = new_total
            self.done = True
            reward = 1
            return (
                np.array(self.player_totals, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Update total and switch turns
            self.player_totals[self.current_player] = new_total
            self.current_player = 1 - self.current_player  # Switch player
            reward = 0
            return (
                np.array(self.player_totals, dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        player1_total, player2_total = self.player_totals
        current_player_str = "Player 1" if self.current_player == 0 else "Player 2"
        render_str = (
            f"Current Totals:\n"
            f"Player 1 = {player1_total}\n"
            f"Player 2 = {player2_total}\n"
        )
        if not self.done:
            render_str += f"It's {current_player_str}'s turn."
        else:
            render_str += "Game over."
        return render_str

    def valid_moves(self):
        # All actions (numbers 1-10) are always valid moves
        return list(range(10))
