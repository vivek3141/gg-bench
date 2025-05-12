import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define primes and action mapping
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        self.num_primes = len(self.primes)

        # Action space: Discrete space of 9 actions (indices of the primes)
        self.action_space = spaces.Discrete(self.num_primes)

        # Observation space: 9 for prime availability, 2 for cumulative scores
        # Primes availability: 0 (not available) or 1 (available)
        # Cumulative scores: integers between 0 and 50
        low_obs = np.zeros(self.num_primes + 2, dtype=np.int32)
        high_obs = np.array([1] * self.num_primes + [50, 50], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.primes_available = np.ones(self.num_primes, dtype=np.int32)
        self.cumulative_scores = np.zeros(
            2, dtype=np.int32
        )  # Player 1 and Player 2 scores
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        self.last_move_turn = [
            0,
            0,
        ]  # Turn number when each player last increased their score
        self.total_turns = 0

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )  # No reward if game is already over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}  # Invalid move

        # Valid move; proceed with updating the game state
        prime_value = self.primes[action]
        self.cumulative_scores[self.current_player] += prime_value
        self.primes_available[action] = 0  # Remove the prime from available pool
        self.total_turns += 1
        self.last_move_turn[self.current_player] = self.total_turns

        # Check for win condition
        if self.cumulative_scores[self.current_player] == 50:
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # If player's cumulative score exceeds 50 (shouldn't happen due to valid_moves check)
        if self.cumulative_scores[self.current_player] > 50:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Check if the pool is exhausted
        if np.sum(self.primes_available) == 0:
            self.done = True
            # Determine the winner
            my_score = self.cumulative_scores[self.current_player]
            opp_score = self.cumulative_scores[1 - self.current_player]

            if my_score > opp_score and my_score < 50:
                reward = 1
            elif my_score == opp_score:
                # Player who reached the high score first wins
                if (
                    self.last_move_turn[self.current_player]
                    <= self.last_move_turn[1 - self.current_player]
                ):
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player = 1 - self.current_player
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        primes_status = [
            f"{self.primes[i]}:{'Available' if self.primes_available[i] == 1 else 'Taken'}"
            for i in range(self.num_primes)
        ]
        primes_str = "Shared Prime Pool:\n" + ", ".join(primes_status) + "\n"
        scores_str = (
            f"Player 1 Score: {self.cumulative_scores[0]}\n"
            f"Player 2 Score: {self.cumulative_scores[1]}\n"
        )
        current_player_str = f"Current Player: Player {self.current_player + 1}\n"
        return primes_str + scores_str + current_player_str

    def valid_moves(self):
        valid_actions = []
        for i in range(self.num_primes):
            if self.primes_available[i] == 1:
                if self.cumulative_scores[self.current_player] + self.primes[i] <= 50:
                    valid_actions.append(i)
        return valid_actions

    def _get_observation(self):
        # Combine the primes availability and the cumulative scores into one observation array
        observation = np.concatenate(
            (self.primes_available, self.cumulative_scores)
        ).astype(np.int32)
        return observation
