import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N  # Starting position
        self.primes = self.generate_primes(self.N)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            len(self.primes)
        )  # Indices of available primes
        self.observation_space = spaces.Box(
            low=1, high=self.N, shape=(2,), dtype=np.int32
        )  # Positions of both players

        self.reset()

    def generate_primes(self, N):
        """Generate a list of prime numbers less than N."""
        primes = []
        for num in range(2, N):
            is_prime = True
            for i in range(2, int(np.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
        return primes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = [self.N, self.N]  # Positions of Player 1 and Player 2
        self.current_player = 0  # Player index: 0 or 1
        self.done = False
        return (
            np.array(self.player_positions, dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return np.array(self.player_positions, dtype=np.int32), 0, True, False, {}

        prime = self.primes[action]
        curr_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[1 - self.current_player]

        # Check if the prime is valid
        if prime >= curr_pos:
            # Invalid move: prime is not less than current position
            reward = -10
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        new_pos = curr_pos - prime

        if new_pos <= 0:
            # Invalid move: new position is less than or equal to 0
            reward = -10
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if new_pos == opponent_pos:
            # Invalid move: new position is occupied by opponent
            reward = -10
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the action is among valid moves
        if action not in self.valid_moves():
            reward = -10
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Update the current player's position
        self.player_positions[self.current_player] = new_pos

        if new_pos == 1:
            # Current player wins
            reward = 1
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Continue the game
            reward = 0
            self.current_player = 1 - self.current_player  # Switch player
            return (
                np.array(self.player_positions, dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        """Return a visual representation of the game state."""
        positions = [f"[{i}]" for i in range(self.N + 1)]  # positions[0] unused

        p0_pos = self.player_positions[0]
        p1_pos = self.player_positions[1]

        # Mark players' positions
        if p0_pos == p1_pos:
            positions[p0_pos] = "[P1&P2]"
        else:
            positions[p0_pos] = "[P1]"
            positions[p1_pos] = "[P2]"

        # Display the path from N down to 1
        path_str = "".join(positions[1:][::-1])  # Exclude index 0 and reverse
        return path_str

    def valid_moves(self):
        """Return a list of valid action indices for the current player."""
        curr_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[1 - self.current_player]

        valid_actions = []
        for idx, prime in enumerate(self.primes):
            if prime >= curr_pos:
                continue  # Prime must be less than current position

            new_pos = curr_pos - prime

            if new_pos <= 0:
                continue  # New position must be greater than 0

            if new_pos == opponent_pos:
                continue  # Cannot move to opponent's position

            valid_actions.append(idx)
        return valid_actions
