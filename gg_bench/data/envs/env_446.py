import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are:
        # 0: Pass
        # 1-8: Selecting factors 2-9
        self.action_space = spaces.Discrete(9)

        # Observation space: current player's number and opponent's number
        # We assume the numbers range from 1 to 1e6
        self.observation_space = spaces.Box(
            low=np.array([1, 1], dtype=np.int32),
            high=np.array([int(1e6), int(1e6)], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize game state
        self.initial_number = 100  # Default initial number
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_numbers = {1: self.initial_number, 2: self.initial_number}
        self.current_player = 1
        self.done = False
        self.consecutive_passes = 0
        # Return observation and info
        observation = np.array(
            [
                self.player_numbers[self.current_player],
                self.player_numbers[self._opponent_player()],
            ],
            dtype=np.int32,
        )
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            raise Exception("Attempted to step environment that is already done.")
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        opponent_player = self._opponent_player()
        current_number = self.player_numbers[self.current_player]
        opponent_number = self.player_numbers[opponent_player]
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            terminated = True
            reward = -10
            observation = np.array([current_number, opponent_number], dtype=np.int32)
            return observation, reward, terminated, truncated, info

        if action == 0:
            # Pass action
            # Only valid if there are no valid factors
            if len(valid_actions) > 1:
                # Passing when there are valid moves is invalid
                self.done = True
                terminated = True
                reward = -10
                observation = np.array(
                    [current_number, opponent_number], dtype=np.int32
                )
                return observation, reward, terminated, truncated, info
            else:
                # Valid pass
                self.consecutive_passes += 1
                if self.consecutive_passes >= 2:
                    # Game ends
                    self.done = True
                    terminated = True
                    if current_number < opponent_number:
                        # Current player wins
                        reward = 1
                    elif current_number > opponent_number:
                        # Current player loses
                        reward = -1
                    else:
                        # Draw
                        reward = 0
                else:
                    # Switch players
                    self.current_player = opponent_player
                observation = np.array(
                    [
                        self.player_numbers[self.current_player],
                        self.player_numbers[self._opponent_player()],
                    ],
                    dtype=np.int32,
                )
                return observation, reward, terminated, truncated, info
        else:
            # Selecting a factor
            factor = action + 1  # Since action 1 corresponds to factor 2
            if opponent_number % factor == 0:
                # Valid move
                self.player_numbers[opponent_player] = opponent_number // factor
                self.consecutive_passes = 0
                # Check for victory
                new_opponent_number = self.player_numbers[opponent_player]
                if new_opponent_number == 1:
                    # Current player wins
                    self.done = True
                    terminated = True
                    reward = 1
                else:
                    # Switch players
                    self.current_player = opponent_player
                observation = np.array(
                    [
                        self.player_numbers[self.current_player],
                        self.player_numbers[self._opponent_player()],
                    ],
                    dtype=np.int32,
                )
                return observation, reward, terminated, truncated, info
            else:
                # Invalid move
                self.done = True
                terminated = True
                reward = -10
                observation = np.array(
                    [current_number, opponent_number], dtype=np.int32
                )
                return observation, reward, terminated, truncated, info

    def render(self):
        opponent_player = self._opponent_player()
        state_str = f"Player {self.current_player}'s turn:\n"
        state_str += f"Your number: {self.player_numbers[self.current_player]}\n"
        state_str += f"Opponent's number: {self.player_numbers[opponent_player]}\n"
        return state_str

    def valid_moves(self):
        opponent_number = self.player_numbers[self._opponent_player()]
        factors = [i for i in range(2, 10) if opponent_number % i == 0]
        if factors:
            valid_actions = [i - 1 for i in factors]
        else:
            valid_actions = [0]  # Only pass is valid
        return valid_actions

    def _opponent_player(self):
        return 2 if self.current_player == 1 else 1
