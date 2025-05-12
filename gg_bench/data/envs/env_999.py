import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(49), actions 0 to 48 correspond to numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Observation space:
        # obs[0]: current player (0 or 1), scaled ((player + 1) / 2)
        # obs[1]: current number / 50.0
        # obs[2:]: availability of numbers 2 to 50
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(51,), dtype=np.float32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.available_numbers = np.ones(49, dtype=np.float32)  # Numbers from 2 to 50
        self.current_player = 1  # 1 or -1
        self.done = False

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        obs = np.zeros(51, dtype=np.float32)
        obs[0] = (self.current_player + 1) / 2.0  # 1 if current_player is 1, 0 if -1
        obs[1] = self.current_number / 50.0
        obs[2:] = self.available_numbers
        return obs

    def valid_moves(self):
        valid_actions = []
        for idx in range(49):
            number = idx + 2  # numbers from 2 to 50
            if self.available_numbers[idx] == 1:
                if (
                    self.current_number % number == 0
                    or number % self.current_number == 0
                ):
                    valid_actions.append(idx)
        return valid_actions

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        number = action + 2  # Map action index to actual number

        valid = False
        if self.available_numbers[action] == 1:
            if self.current_number % number == 0 or number % self.current_number == 0:
                valid = True

        if not valid:
            # Invalid move
            reward = -10.0
            done = False
        else:
            # Valid move
            self.current_number = number
            self.available_numbers[action] = 0  # Remove number from pool

            # Check if opponent has any valid moves
            opponent_valid_moves = []
            for idx in range(49):
                opp_number = idx + 2
                if self.available_numbers[idx] == 1:
                    if (
                        self.current_number % opp_number == 0
                        or opp_number % self.current_number == 0
                    ):
                        opponent_valid_moves.append(idx)

            if not opponent_valid_moves:
                # Opponent cannot move, current player wins
                reward = 1.0
                done = True
            else:
                # Game continues
                reward = 0.0
                done = False

        if not done:
            # Switch player
            self.current_player *= -1

        obs = self._get_observation()
        self.done = done
        return obs, reward, done, False, {}

    def render(self):
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        available_numbers = [
            idx + 2 for idx, val in enumerate(self.available_numbers) if val == 1
        ]
        available_numbers_str = ", ".join(map(str, available_numbers))
        render_str = f"Current Player: {player_str}\nCurrent Number: {self.current_number}\nAvailable Numbers: {available_numbers_str}\n"
        print(render_str)
