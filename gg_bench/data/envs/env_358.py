import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the primes and action mapping
        self.primes = [2, 3, 5, 7, 11, 13]
        self.action_space = spaces.Discrete(len(self.primes))

        # Observation space:
        # [Player 1 position (0-100), Player 2 position (0-100),
        #  Player 1 last prime used (-1 to 5), Player 2 last prime used (-1 to 5),
        #  Current player (0 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0]),
            high=np.array([100, 100, len(self.primes) - 1, len(self.primes) - 1, 1]),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = [0, 0]  # Positions of Player 1 and Player 2
        self.last_primes_used = [-1, -1]  # Indices of last primes used (-1 means none)
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        prime_index = action
        prime = self.primes[prime_index]
        current_player = self.current_player
        opponent = 1 - current_player
        last_prime_index = self.last_primes_used[current_player]
        current_position = self.player_positions[current_player]

        # Check if the move is valid
        if last_prime_index == prime_index:
            # Cannot reuse the same prime as the last turn
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        new_position = current_position + prime
        if new_position > 100:
            # Cannot exceed position 100
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Update the game state
        self.player_positions[current_player] = new_position
        self.last_primes_used[current_player] = prime_index

        # Check for win condition
        if new_position == 100:
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Switch current player
        self.current_player = opponent
        reward = 0
        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        s = f"Player 1 Position: {self.player_positions[0]}, Last Prime Used: "
        if self.last_primes_used[0] == -1:
            s += "None\n"
        else:
            s += f"{self.primes[self.last_primes_used[0]]}\n"
        s += f"Player 2 Position: {self.player_positions[1]}, Last Prime Used: "
        if self.last_primes_used[1] == -1:
            s += "None\n"
        else:
            s += f"{self.primes[self.last_primes_used[1]]}\n"
        s += f"Current Player: {1 if self.current_player == 0 else 2}\n"
        return s

    def valid_moves(self):
        valid_actions = []
        current_player = self.current_player
        last_prime_index = self.last_primes_used[current_player]
        current_position = self.player_positions[current_player]
        for index, prime in enumerate(self.primes):
            if index == last_prime_index:
                continue  # Cannot choose the same prime as last turn
            if current_position + prime > 100:
                continue  # Cannot exceed position 100
            valid_actions.append(index)
        return valid_actions

    def _get_observation(self):
        return np.array(
            [
                self.player_positions[0],
                self.player_positions[1],
                self.last_primes_used[0],
                self.last_primes_used[1],
                self.current_player,
            ],
            dtype=np.int32,
        )
