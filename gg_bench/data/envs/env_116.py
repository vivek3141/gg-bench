import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Digits 1-9
        self.action_space = spaces.Discrete(
            9
        )  # Actions are indices 0-8, corresponding to digits 1-9

        # Observation space: [shared number, player1 LP, player2 LP]
        # Shared number ranges from -1000 to 1000 for practical purposes
        high = np.array([1000, 10, 10], dtype=np.int32)
        low = np.array([-1000, 0, 0], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 0
        self.player1_lp = 10
        self.player2_lp = 10
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Return initial observation and info
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game already over

        # Map action index to digit (1-9)
        digit = action + 1  # Action 0 corresponds to digit 1, action 8 to digit 9

        # Determine operation based on shared number parity
        if self.shared_number % 2 == 0:
            # Even shared number: add digit
            new_shared_number = self.shared_number + digit
        else:
            # Odd shared number: subtract digit
            new_shared_number = self.shared_number - digit

        # Initialize penalties and reset flag
        penalty = 0  # Penalty to active player's Life Points
        opponent_penalty = 0  # Penalty to opponent's Life Points
        reset_shared_number = None

        # Apply special conditions and penalties
        if new_shared_number < 0:
            # Negative shared number: active player loses 1 Life Point
            penalty += 1
        if new_shared_number == 15:
            # Shared number is 15: opponent loses 3 Life Points, reset shared number to 0
            opponent_penalty += 3
            reset_shared_number = 0
        if new_shared_number > 20:
            # Shared number exceeds 20: active player loses 2 Life Points, reset shared number to 10
            penalty += 2
            reset_shared_number = 10

        # Update shared number
        if reset_shared_number is not None:
            self.shared_number = reset_shared_number
        else:
            self.shared_number = new_shared_number

        # Update Life Points
        if self.current_player == 1:
            self.player1_lp -= penalty
            self.player2_lp -= opponent_penalty
            active_player_lp = self.player1_lp
            opponent_lp = self.player2_lp
        else:
            self.player2_lp -= penalty
            self.player1_lp -= opponent_penalty
            active_player_lp = self.player2_lp
            opponent_lp = self.player1_lp

        # Check for victory conditions
        if opponent_lp <= 0:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        elif active_player_lp <= 0:
            # Current player loses
            self.done = True
            reward = -1
            return self._get_obs(), reward, True, False, {}
        else:
            # Valid move with no victory
            reward = -10  # Negative reward for each valid move

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return observation, reward, done flag, truncated flag, and info
        return self._get_obs(), reward, False, False, {}

    def render(self):
        # Create a string representation of the current game state
        s = f"Player 1 Life Points: {self.player1_lp}\n"
        s += f"Player 2 Life Points: {self.player2_lp}\n"
        s += f"Current Number: {self.shared_number}\n"
        s += f"Player {self.current_player}'s Turn\n"
        return s

    def valid_moves(self):
        # Return all possible actions (digits 1-9 represented as indices 0-8)
        return list(range(9))

    def _get_obs(self):
        # Helper function to get current observation
        return np.array(
            [self.shared_number, self.player1_lp, self.player2_lp], dtype=np.int32
        )
