import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-4 => Attack with values 1-5, 5 => Defend
        self.action_space = spaces.Discrete(6)

        # Define observation space: [Current Player SP, Opponent SP]
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Initialize state variables
        self.sp = None  # Shield Points for both players
        self.current_player = None  # Current player's index (0 or 1)
        self.done = None  # Game over flag

        # Reset the environment to start state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sp = [10, 10]  # Both players start with 10 SP
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Observation: [Current Player's SP, Opponent's SP]
        return np.array(
            [self.sp[self.current_player], self.sp[1 - self.current_player]],
            dtype=np.int32,
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, self.done, False, {}

        if action == 5:
            # Defend action
            self.sp[self.current_player] += 1
            reward = 0
        else:
            # Attack action
            attack_value = action + 1  # Map action 0-4 to attack value 1-5
            opponent = 1 - self.current_player
            self.sp[opponent] -= attack_value
            if self.sp[opponent] <= 0:
                # Current player wins
                self.done = True
                return self._get_obs(), 1, self.done, False, {}
            reward = 0

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        # Visual representation of the game state
        output = f"Player {self.current_player + 1}'s turn.\n"
        output += f"Player 1 SP: {self.sp[0]}\n"
        output += f"Player 2 SP: {self.sp[1]}\n"
        return output

    def valid_moves(self):
        # Valid actions for the current player
        moves = [0, 1, 2, 3, 4, 5]  # All possible actions
        if self.sp[self.current_player] >= 10:
            moves.remove(5)  # Cannot defend if SP is at maximum
        return moves
