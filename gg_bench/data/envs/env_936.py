import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 20 numbers to choose from (1 to 20)
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # - Number pool: 20 positions (numbers 1-20), value 0 (taken) or actual number (available)
        # - Current player's numbers: 5 positions, numbers selected or 0
        # - Opponent's numbers: 5 positions, numbers selected or 0
        # Total observation size: 20 + 5 + 5 = 30
        self.observation_space = spaces.Box(low=0, high=20, shape=(30,), dtype=np.int32)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool: numbers from 1 to 20 are available
        self.number_pool = np.ones(20, dtype=np.int32)

        # Initialize players' selected numbers
        self.player1_numbers = np.zeros(5, dtype=np.int32)
        self.player2_numbers = np.zeros(5, dtype=np.int32)

        # Assign secret criteria
        self.player1_criterion = self._assign_random_criterion()
        self.player2_criterion = self._assign_random_criterion()
        while self.player2_criterion == self.player1_criterion:
            self.player2_criterion = self._assign_random_criterion()

        # Initialize current player and game status
        self.current_player = 1
        self.done = False
        self.info = {}

        observation = self._get_observation()
        return observation, self.info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, self.info

        if action < 0 or action >= 20:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, self.info

        if self.number_pool[action] == 0:
            # Number already taken
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, self.info

        # Valid action
        number_selected = action + 1  # Numbers are from 1 to 20

        # Remove the number from the pool
        self.number_pool[action] = 0

        # Add the number to the current player's collection
        if self.current_player == 1:
            player_numbers = self.player1_numbers
            player_criterion = self.player1_criterion
        else:
            player_numbers = self.player2_numbers
            player_criterion = self.player2_criterion

        idx = np.where(player_numbers == 0)[0]
        if len(idx) == 0:
            # Player's collection is full
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, self.info
        else:
            player_numbers[idx[0]] = number_selected

        # Check for victory condition
        qualifying_numbers = self._get_qualifying_numbers(
            player_numbers, player_criterion
        )
        if len(qualifying_numbers) >= 3:
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, self.info

        # Check if all numbers have been selected
        if np.sum(self.number_pool) == 0:
            # Game end, determine winner
            player1_qualifying = len(
                self._get_qualifying_numbers(
                    self.player1_numbers, self.player1_criterion
                )
            )
            player2_qualifying = len(
                self._get_qualifying_numbers(
                    self.player2_numbers, self.player2_criterion
                )
            )
            if player1_qualifying > player2_qualifying:
                winner = 1
            elif player2_qualifying > player1_qualifying:
                winner = 2
            else:
                # Tie game
                reward = 0
                self.done = True
                return self._get_observation(), reward, True, False, self.info
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1
            self.done = True
            return self._get_observation(), reward, True, False, self.info

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return self._get_observation(), reward, False, False, self.info

    def render(self):
        s = f"Number pool: {[i+1 for i in range(20) if self.number_pool[i]==1]}\n"
        s += f"Player 1 numbers: {list(self.player1_numbers[self.player1_numbers != 0])}\n"
        s += f"Player 2 numbers: {list(self.player2_numbers[self.player2_numbers != 0])}\n"
        s += f"Current player: Player {self.current_player}\n"
        return s

    def valid_moves(self):
        return [i for i in range(20) if self.number_pool[i] == 1]

    def _assign_random_criterion(self):
        criteria_list = [
            "Even",
            "Odd",
            "Prime",
            "MultipleOf3",
            "PerfectSquare",
            "Fibonacci",
        ]
        return self.np_random.choice(criteria_list)

    def _get_observation(self):
        number_pool_obs = np.where(self.number_pool == 1, np.arange(1, 21), 0)

        if self.current_player == 1:
            own_numbers = self.player1_numbers
            opp_numbers = self.player2_numbers
        else:
            own_numbers = self.player2_numbers
            opp_numbers = self.player1_numbers

        observation = np.concatenate([number_pool_obs, own_numbers, opp_numbers])
        return observation

    def _get_qualifying_numbers(self, numbers, criterion):
        criteria_functions = {
            "Even": self._is_even,
            "Odd": self._is_odd,
            "Prime": self._is_prime,
            "MultipleOf3": self._is_multiple_of_3,
            "PerfectSquare": self._is_perfect_square,
            "Fibonacci": self._is_fibonacci,
        }
        func = criteria_functions[criterion]
        qualifying_numbers = [n for n in numbers if n != 0 and func(n)]
        return qualifying_numbers

    def _is_even(self, n):
        return n % 2 == 0

    def _is_odd(self, n):
        return n % 2 == 1

    def _is_prime(self, n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _is_multiple_of_3(self, n):
        return n % 3 == 0

    def _is_perfect_square(self, n):
        return int(np.sqrt(n)) ** 2 == n

    def _is_fibonacci(self, n):
        return self._is_perfect_square(5 * n * n + 4) or self._is_perfect_square(
            5 * n * n - 4
        )
