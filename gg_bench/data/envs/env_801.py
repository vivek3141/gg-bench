import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Numbers 1-9 represented by indices 0-8

        # Observation space consists of:
        # - Number pool: 9 values (0 or 1)
        # - Player cumulative scores: 2 values
        # - Clue code: 1 value
        # Total: 12 values
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [0, 0, 0]),
            high=np.array([1] * 9 + [45, 45, 3]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Randomly select target sum between 15 and 25 inclusive
        self.target_sum = self.np_random.randint(15, 26)

        # Initialize number pool (1 if available, 0 if taken)
        self.number_pool = np.ones(9, dtype=np.int32)

        # Initialize cumulative scores
        self.cumulative_scores = [0, 0]  # [Player 1 score, Player 2 score]

        # Start with Player 1
        self.current_player = 0  # 0 for Player 1, 1 for Player 2

        # Initialize the clue code to 0
        self.clue_code = 0

        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Perform action
        selected_number = action + 1  # Map index to actual number

        # Remove number from pool
        self.number_pool[action] = 0

        # Update cumulative score
        self.cumulative_scores[self.current_player] += selected_number

        # Provide clue
        player_score = self.cumulative_scores[self.current_player]
        difference = self.target_sum - player_score

        if player_score == self.target_sum:
            self.clue_code = 2  # "Your cumulative score equals the target sum!"
        elif player_score > self.target_sum:
            self.clue_code = 3  # "Your cumulative score exceeds the target sum."
        elif difference <= 5:
            self.clue_code = 1  # "Your cumulative score is within 5 of the target sum."
        else:
            self.clue_code = (
                0  # "Your cumulative score is less than the target sum by more than 5."
            )

        # Check for win/loss conditions
        if player_score == self.target_sum:
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}
        elif player_score > self.target_sum:
            # Current player loses
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}
        elif np.sum(self.number_pool) == 0:
            # All numbers exhausted
            p1_diff = (
                self.target_sum - self.cumulative_scores[0]
                if self.cumulative_scores[0] <= self.target_sum
                else -np.inf
            )
            p2_diff = (
                self.target_sum - self.cumulative_scores[1]
                if self.cumulative_scores[1] <= self.target_sum
                else -np.inf
            )

            if p1_diff > p2_diff:
                winner = 0
            elif p2_diff > p1_diff:
                winner = 1
            else:
                # Tie, sudden death not implemented, declare draw
                winner = None

            self.done = True
            if winner == self.current_player:
                reward = 1
            else:
                reward = -10
            return self._get_observation(), reward, True, False, {}
        else:
            # Continue game
            reward = 0
            # Switch to next player
            self.current_player = 1 - self.current_player
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Prepare visual representation of the game state
        number_pool_str = "Available numbers: " + ",".join(
            [str(i + 1) for i in range(9) if self.number_pool[i] == 1]
        )
        player_scores_str = f"Player 1 Score: {self.cumulative_scores[0]}, Player 2 Score: {self.cumulative_scores[1]}"
        clues = [
            "Your cumulative score is less than the target sum by more than 5.",
            "Your cumulative score is within 5 of the target sum.",
            "Your cumulative score equals the target sum!",
            "Your cumulative score exceeds the target sum.",
        ]
        clue_str = f"Clue for Player {self.current_player + 1}: {clues[self.clue_code]}"
        return f"{number_pool_str}\n{player_scores_str}\n{clue_str}\n"

    def valid_moves(self):
        return [i for i in range(9) if self.number_pool[i] == 1]

    def _get_observation(self):
        observation = np.concatenate(
            (
                self.number_pool,
                np.array(self.cumulative_scores, dtype=np.int32),
                np.array([self.clue_code], dtype=np.int32),
            )
        )
        return observation
