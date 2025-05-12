import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers from 1 to 50 (actions 0 to 49)
        self.action_space = spaces.Discrete(50)

        # Observation space:
        # First 50 elements: number pool status (-1: opponent used, 0: unused, 1: current player used)
        # Next element: last number in current player's chain (0 if empty)
        # Next element: last number in opponent's chain (0 if empty)
        # Next element: current player indicator (1 or -1)
        self.observation_space = spaces.Box(
            low=np.array([-1] * 50 + [0, 0, -1]),
            high=np.array([1] * 50 + [50, 50, 1]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.zeros(
            50, dtype=np.int32
        )  # 0: unused, 1: used by player 1, -1: used by player -1
        self.last_numbers = {1: 0, -1: 0}  # Last numbers for each player
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            raise Exception("Game is already over")

        number = action + 1  # Map action index to actual number

        if action < 0 or action >= 50:
            raise ValueError("Invalid action: should be between 0 and 49")

        # Check if the number is in the number pool and unused
        if self.number_pool[action] != 0:
            # Invalid move
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Check if the number satisfies the chain rule
        last_number = self.last_numbers[self.current_player]

        if last_number == 0:
            # First number can be any number
            pass
        else:
            # Check if the number is a divisor or multiple of last_number
            if not (last_number % number == 0 or number % last_number == 0):
                # Invalid move
                reward = -10
                self.done = True
                observation = self._get_observation()
                return observation, reward, self.done, False, {}

        # Valid move
        # Update number pool
        self.number_pool[action] = self.current_player

        # Update last number for current player
        self.last_numbers[self.current_player] = number

        # Check if the next player has any valid moves
        next_player = -self.current_player

        valid_moves = self._get_valid_moves(next_player)

        if not valid_moves:
            # Next player cannot move, current player wins
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        else:
            # Game continues
            self.current_player = next_player
            reward = 0
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

    def render(self):
        number_pool_status = "".join(
            ["." if x == 0 else ("X" if x == 1 else "O") for x in self.number_pool]
        )
        chain_player1 = [i + 1 for i, x in enumerate(self.number_pool) if x == 1]
        chain_player2 = [i + 1 for i, x in enumerate(self.number_pool) if x == -1]
        render_str = (
            f"Number Pool (unused numbers marked as '.'): {number_pool_status}\n"
            f"Player 1 Chain: {chain_player1}\n"
            f"Player 2 Chain: {chain_player2}\n"
            f"Current Player: {'Player 1' if self.current_player ==1 else 'Player 2'}\n"
        )
        print(render_str)

    def valid_moves(self):
        return self._get_valid_moves(self.current_player)

    def _get_valid_moves(self, player):
        last_number = self.last_numbers[player]
        valid_moves = []
        for idx in range(50):
            if self.number_pool[idx] == 0:
                number = idx + 1
                if last_number == 0:
                    valid_moves.append(idx)
                elif last_number % number == 0 or number % last_number == 0:
                    valid_moves.append(idx)
        return valid_moves

    def _get_observation(self):
        obs = np.concatenate(
            [
                self.number_pool,
                np.array(
                    [
                        self.last_numbers[self.current_player],
                        self.last_numbers[-self.current_player],
                        self.current_player,
                    ]
                ),
            ]
        )
        return obs.astype(np.int32)
