import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
        ]  # Prime numbers less than 50

        self.num_primes = len(self.primes)
        self.action_space = spaces.Discrete(self.num_primes)

        # Observation space:
        # Index 0: Current player's LP
        # Index 1: Opponent's LP
        # Index 2-16: Prime availability (1 = available, 0 = used)
        self.observation_space = spaces.Box(
            low=0,
            high=50,
            shape=(2 + self.num_primes,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_LP = [50, 50]  # [Player 0's LP, Player 1's LP]
        self.prime_available = np.ones(self.num_primes, dtype=np.int32)
        self.current_player = 0  # Player 0 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Get the prime number corresponding to the action
        prime = self.primes[action]
        opponent = 1 - self.current_player

        # Apply attack
        self.player_LP[opponent] -= prime

        # Remove the prime from availability
        self.prime_available[action] = 0

        # Check for victory
        if self.player_LP[opponent] <= 0:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch current player
        self.current_player = opponent

        observation = self._get_observation()
        return observation, 0, False, False, {}  # Continue game

    def render(self):
        current_player_lp = self.player_LP[self.current_player]
        opponent_player_lp = self.player_LP[1 - self.current_player]
        available_primes = [
            prime
            for prime, available in zip(self.primes, self.prime_available)
            if available
        ]

        render_str = "---------------------------------------\n"
        render_str += f"Player {self.current_player + 1}'s Turn\n"
        render_str += "---------------------------------------\n"
        render_str += f"Your LP: {current_player_lp}\n"
        render_str += f"Opponent's LP: {opponent_player_lp}\n"
        render_str += f"Available Primes: {available_primes}\n"
        return render_str

    def valid_moves(self):
        # Returns a list of valid action indices
        opponent_lp = self.player_LP[1 - self.current_player]
        available_indices = np.where(self.prime_available == 1)[0]

        # Primes less than or equal to opponent's LP
        valid_indices = [
            idx for idx in available_indices if self.primes[idx] <= opponent_lp
        ]

        if valid_indices:
            return valid_indices
        else:
            # If no such prime is available, select the smallest available prime
            if available_indices.size > 0:
                min_prime_idx = available_indices[
                    np.argmin([self.primes[idx] for idx in available_indices])
                ]
                return [min_prime_idx]
            else:
                return []  # No primes left

    def _get_observation(self):
        # Construct the observation array
        observation = np.zeros(2 + self.num_primes, dtype=np.int32)
        observation[0] = self.player_LP[self.current_player]
        observation[1] = self.player_LP[1 - self.current_player]
        observation[2:] = self.prime_available
        return observation
