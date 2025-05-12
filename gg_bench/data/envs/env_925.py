import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            99
        )  # Actions: numbers from 1 to 99 (indices 0 to 98)

        # Observation space: Number pool (99 numbers) + Required digit one-hot (10 digits)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(109,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool: 1 if the number is available, 0 if taken
        self.number_pool = np.ones(99, dtype=np.float32)
        self.current_number = None  # No current number at the start
        self.required_digit = None  # No required digit at the start
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.last_action = None
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number = action + 1  # Actions are mapped to numbers 1 to 99

        # Check if the number is available in the number pool
        if self.number_pool[action] == 0:
            return self._invalid_move()

        # Check if the move is valid
        if self.required_digit is not None:
            first_digit = int(str(number)[0])
            if first_digit != self.required_digit:
                return self._invalid_move()

        # Valid move
        self.number_pool[action] = 0  # Remove the number from the pool
        self.current_number = number
        self.last_action = action
        self.required_digit = int(
            str(number)[-1]
        )  # Update required digit for the next player

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Current player wins
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        pool_numbers = [str(i + 1) for i in range(99) if self.number_pool[i] == 1]
        pool_str = ", ".join(pool_numbers)
        if self.required_digit is not None:
            required_digit_str = f"Required first digit: {self.required_digit}"
        else:
            required_digit_str = "No required digit (first move)"
        render_str = f"Available Numbers: {pool_str}\n{required_digit_str}"
        return render_str

    def valid_moves(self):
        valid_actions = []
        for i in range(99):
            if self.number_pool[i] == 1:
                number = i + 1
                if self.required_digit is None:
                    valid_actions.append(i)
                else:
                    first_digit = int(str(number)[0])
                    if first_digit == self.required_digit:
                        valid_actions.append(i)
        return valid_actions

    def _invalid_move(self):
        # Invalid move: the player loses
        self.done = True
        reward = -10
        return self._get_observation(), reward, True, False, {}

    def _get_observation(self):
        obs = np.zeros(109, dtype=np.float32)
        obs[:99] = self.number_pool
        if self.required_digit is not None:
            obs[99 + self.required_digit] = 1
        return obs
