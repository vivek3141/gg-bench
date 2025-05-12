import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - Counts of numbers 1-9 in the pool (9 values)
        # - Current player's cumulative sum (1 value)
        # - Opponent's cumulative sum (1 value)
        # - Target sum (1 value)
        self.observation_space = spaces.Box(
            low=0,
            high=25,
            shape=(9 + 3,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the target sum between 15 and 25
        self.target_sum = self.np_random.integers(15, 26)
        # Initialize the pool with two copies of numbers 1-9
        self.pool_counts = np.array([2] * 9, dtype=np.int32)
        # Initialize player sums
        self.player_sums = [0, 0]  # Index 0: Player 1, Index 1: Player 2
        # Set the current player (0 or 1)
        self.current_player = 0
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number_selected = action + 1  # Actions 0-8 correspond to numbers 1-9

        if self.pool_counts[action] <= 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        self.pool_counts[action] -= 1
        self.player_sums[self.current_player] += number_selected

        # Check for win/loss conditions
        player_sum = self.player_sums[self.current_player]

        if player_sum == self.target_sum:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        elif player_sum > self.target_sum:
            # Current player loses
            self.done = True
            return self._get_observation(), -10, True, False, {}
        elif np.sum(self.pool_counts) == 0:
            # Pool is exhausted
            other_player = 1 - self.current_player
            other_player_sum = self.player_sums[other_player]

            if player_sum > self.target_sum and other_player_sum > self.target_sum:
                # Both players exceeded target sum
                self.done = True
                return self._get_observation(), -10, True, False, {}
            elif player_sum > self.target_sum:
                # Current player exceeded target sum
                self.done = True
                return self._get_observation(), -10, True, False, {}
            elif other_player_sum > self.target_sum:
                # Opponent exceeded target sum
                self.done = True
                return self._get_observation(), 1, True, False, {}
            else:
                # Evaluate who is closer to the target sum
                curr_diff = self.target_sum - player_sum
                opp_diff = self.target_sum - other_player_sum

                if curr_diff < opp_diff:
                    self.done = True
                    return self._get_observation(), 1, True, False, {}
                else:
                    self.done = True
                    return self._get_observation(), -10, True, False, {}
        else:
            # Switch to the other player
            self.current_player = 1 - self.current_player
            return self._get_observation(), -10, False, False, {}

    def render(self):
        pool_str = "Available Numbers: "
        for i in range(9):
            pool_str += f"{i+1}({self.pool_counts[i]}), "
        pool_str = pool_str.rstrip(", ")
        curr_player = "Player 1" if self.current_player == 0 else "Player 2"
        render_str = f"Target Sum: {self.target_sum}\n"
        render_str += pool_str + "\n"
        render_str += f"{curr_player}'s turn.\n"
        render_str += f"Player 1 Sum: {self.player_sums[0]}\n"
        render_str += f"Player 2 Sum: {self.player_sums[1]}\n"
        return render_str

    def valid_moves(self):
        return [i for i in range(9) if self.pool_counts[i] > 0]

    def _get_observation(self):
        obs = np.concatenate(
            (
                self.pool_counts.copy(),  # Counts of numbers 1-9 in the pool
                np.array(
                    [
                        self.player_sums[self.current_player],  # Current player's sum
                        self.player_sums[1 - self.current_player],  # Opponent's sum
                        self.target_sum,  # Target sum
                    ],
                    dtype=np.int32,
                ),
            )
        )
        return obs
