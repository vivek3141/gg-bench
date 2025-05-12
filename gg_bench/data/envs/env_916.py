import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space: an array of size 9
        # 0: number is in the pool
        # 1: number is in current player's collection
        # -1: number is in opponent's collection
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self.observation.copy(), {}

    def step(self, action):
        if self.observation[action] != 0 or self.done:
            # Invalid move
            return self.observation.copy(), -10, True, False, {}

        # Valid move
        self.observation[action] = self.current_player

        # Check if current player has won
        player_numbers = [
            i + 1 for i, v in enumerate(self.observation) if v == self.current_player
        ]
        if len(player_numbers) >= 3:
            # Check all combinations of 3 numbers for sum == 15
            for combo in combinations(player_numbers, 3):
                if sum(combo) == 15:
                    self.done = True
                    return self.observation.copy(), 1, True, False, {}

        # Check if all numbers have been selected
        if np.all(self.observation != 0):
            self.done = True
            # Determine the winner based on closest sum to 15 without exceeding it
            opponent_numbers = [
                i + 1
                for i, v in enumerate(self.observation)
                if v == -self.current_player
            ]

            player_sum = sum(player_numbers)
            opponent_sum = sum(opponent_numbers)

            player_closest = player_sum if player_sum <= 15 else 0
            opponent_closest = opponent_sum if opponent_sum <= 15 else 0

            if player_closest > opponent_closest:
                reward = 1
            elif player_closest < opponent_closest:
                reward = -1
            else:
                # Tie-breaker: Lowest individual number
                player_lowest = min(player_numbers) if player_numbers else 10
                opponent_lowest = min(opponent_numbers) if opponent_numbers else 10
                if player_lowest < opponent_lowest:
                    reward = 1
                else:
                    reward = -1  # Opponent wins

            return self.observation.copy(), reward, True, False, {}

        # Switch to next player
        self.current_player *= -1
        return self.observation.copy(), 0, False, False, {}

    def render(self):
        pool_numbers = [str(i + 1) for i, v in enumerate(self.observation) if v == 0]
        player1_numbers = [str(i + 1) for i, v in enumerate(self.observation) if v == 1]
        player2_numbers = [
            str(i + 1) for i, v in enumerate(self.observation) if v == -1
        ]

        render_str = f"Number Pool: {' '.join(pool_numbers)}\n"
        render_str += f"Player 1 Collection: {' '.join(player1_numbers)}\n"
        render_str += f"Player 2 Collection: {' '.join(player2_numbers)}\n"

        return render_str

    def valid_moves(self):
        return [i for i in range(9) if self.observation[i] == 0]
