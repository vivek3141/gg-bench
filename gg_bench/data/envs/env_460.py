import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_score=20, M=10):
        super(CustomEnv, self).__init__()

        self.target_score = target_score  # Target score T
        self.M = M  # Maximum number M
        self.numbers = np.arange(1, self.M + 1)  # Numbers from 1 to M

        # Define action space: actions correspond to indices of numbers from 1 to M
        self.action_space = spaces.Discrete(self.M)

        # Observation space:
        # - First M elements: availability of numbers (1 if available, 0 if not)
        # - Next two elements: current player's score, opponent's score
        # Observation shape: (M + 2,)
        low = np.array([0] * self.M + [0, 0], dtype=np.float32)
        high = np.array(
            [1] * self.M + [self.target_score * 2, self.target_score * 2],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            self.M, dtype=np.float32
        )  # All numbers are initially available
        self.player_scores = {1: 0, -1: 0}
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game over flag
        self.last_action = None  # Last action taken

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )

        number = action + 1  # Map action to number (1 to M)
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Check if action is valid
        is_available = self.available_numbers[action] == 1
        parity_valid = self._is_parity_valid(number)

        if not is_available or not parity_valid:
            # Invalid move
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Valid move
        self.available_numbers[action] = 0  # Mark the number as used
        self.player_scores[self.current_player] += number
        self.last_action = action

        # Check for win
        if self.player_scores[self.current_player] == self.target_score:
            reward = 1
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Check for game end (no valid moves for both players)
        if not self._has_valid_moves():
            # Determine winner
            opponent = -self.current_player
            current_score = self.player_scores[self.current_player]
            opponent_score = self.player_scores[opponent]

            if (
                current_score <= self.target_score
                and opponent_score <= self.target_score
            ):
                if current_score > opponent_score:
                    reward = 1  # Current player wins
                else:
                    reward = -1  # Current player loses
            else:
                # Scores exceeded target
                if (
                    current_score > self.target_score
                    and opponent_score > self.target_score
                ):
                    if current_score < opponent_score:
                        reward = 1  # Current player wins
                    else:
                        reward = -1  # Current player loses
                elif current_score <= self.target_score:
                    reward = 1  # Current player wins
                else:
                    reward = -1  # Current player loses

            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Swap current player
        self.current_player *= -1

        # Return observation
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        available_numbers_str = ", ".join(
            str(i + 1) for i in range(self.M) if self.available_numbers[i] == 1
        )
        state_str = f"Target Score (T): {self.target_score}\n"
        state_str += f"Available Numbers: [{available_numbers_str}]\n\n"
        state_str += f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
        state_str += f"Player 1 Score: {self.player_scores[1]}\n"
        state_str += f"Player 2 Score: {self.player_scores[-1]}\n"
        if self.last_action is not None:
            last_player = 2 if self.current_player == 1 else 1
            last_number = self.last_action + 1
            state_str += f"Player {last_player} selected number {last_number}.\n"
        return state_str

    def valid_moves(self):
        # Returns a list of valid actions (indices) for the current player
        valid_actions = []
        current_parity = self.player_scores[self.current_player] % 2
        for action in range(self.M):
            if self.available_numbers[action] == 1:
                number = action + 1
                if self._is_parity_valid(number):
                    valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Observation includes available numbers and both players' scores
        observation = np.concatenate(
            (
                self.available_numbers,
                np.array(
                    [
                        self.player_scores[self.current_player],
                        self.player_scores[-self.current_player],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        return observation

    def _is_parity_valid(self, number):
        # Checks if the selected number complies with the Parity Rule
        current_score = self.player_scores[self.current_player]
        if current_score % 2 == 0:
            # Current score is even; must pick an odd number
            return number % 2 == 1
        else:
            # Current score is odd; must pick an even number
            return number % 2 == 0

    def _has_valid_moves(self):
        # Checks if there are valid moves for either player
        for player in [self.current_player, -self.current_player]:
            current_parity = self.player_scores[player] % 2
            for action in range(self.M):
                if self.available_numbers[action] == 1:
                    number = action + 1
                    if (current_parity == 0 and number % 2 == 1) or (
                        current_parity == 1 and number % 2 == 0
                    ):
                        return True
        return False
