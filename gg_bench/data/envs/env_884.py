import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Fixed list of prime numbers and mapping of indices to primes
        self.primes_list = [2, 3, 5, 7, 11, 13, 17, 19]
        # Define action space: indices 0-7 correspond to primes, 8 corresponds to 'pass'
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # Observation includes:
        # - Player 1 position (0-25)
        # - Player 2 position (0-25)
        # - Primes available (0 or 1 for each)
        self.observation_space = spaces.Box(low=0, high=25, shape=(10,), dtype=np.int32)
        # [pos_player1, pos_player2, primes_available[8]]

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos_player1 = 0  # Player 1 starting position
        self.pos_player2 = 0  # Player 2 starting position
        self.primes_available = [1] * 8  # All primes are initially available
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.winner = None
        observation = self._get_obs()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Get the list of valid actions for the current player
        valid_actions = self.valid_moves()

        # Handle 'pass' action (action index 8)
        if action == 8:
            if action in valid_actions:
                # Valid pass action
                self.current_player = 2 if self.current_player == 1 else 1
                return self._get_obs(), 0, False, False, {}
            else:
                # Invalid pass (attempting to pass when moves are available)
                self.done = True
                return self._get_obs(), -10, True, False, {}

        # Check if action is valid
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Retrieve the prime number corresponding to the action
        prime = self.primes_list[action]

        # Update the primes pool
        self.primes_available[action] = 0  # Remove the chosen prime

        # Update the current player's position
        if self.current_player == 1:
            self.pos_player1 += prime
            curr_pos = self.pos_player1
            opp_pos = self.pos_player2
        else:
            self.pos_player2 += prime
            curr_pos = self.pos_player2
            opp_pos = self.pos_player1

        # Check if move exceeds position 25 (should not happen)
        if curr_pos > 25:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check for landing on the opponent
        if curr_pos == opp_pos and curr_pos != 0:
            # Send opponent back to start
            if self.current_player == 1:
                self.pos_player2 = 0
            else:
                self.pos_player1 = 0

        # Check for a win
        if curr_pos == 25:
            self.done = True
            self.winner = self.current_player
            return self._get_obs(), 1, True, False, {}  # Reward for winning

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Generate a visual representation of the game state
        path = ["_" for _ in range(25)]
        if self.pos_player1 > 0:
            path[self.pos_player1 - 1] = "1"
        if self.pos_player2 > 0:
            if path[self.pos_player2 - 1] == "_":
                path[self.pos_player2 - 1] = "2"
            elif path[self.pos_player2 - 1] == "1":
                path[self.pos_player2 - 1] = "B"  # Both players on same position
        path_str = " ".join(path)
        primes_available_str = ", ".join(
            [str(self.primes_list[i]) for i in range(8) if self.primes_available[i]]
        )
        return (
            f"Current Player: {self.current_player}\n"
            f"Path: {path_str}\n"
            f"Available Primes: {primes_available_str}\n"
            f"Player 1 Position: {self.pos_player1}\n"
            f"Player 2 Position: {self.pos_player2}\n"
        )

    def valid_moves(self):
        # Determine the valid actions for the current player
        valid_actions = []
        if self.done:
            return valid_actions
        current_pos = self.pos_player1 if self.current_player == 1 else self.pos_player2
        for i, available in enumerate(self.primes_available):
            if available:
                prime = self.primes_list[i]
                new_pos = current_pos + prime
                if new_pos <= 25:
                    valid_actions.append(i)
        if not valid_actions:
            # If no valid moves, 'pass' action is valid
            valid_actions.append(8)  # Action index 8 represents 'pass'
        return valid_actions

    def _get_obs(self):
        # Construct the observation array
        observation = np.array(
            [self.pos_player1, self.pos_player2] + self.primes_available, dtype=np.int32
        )
        return observation
