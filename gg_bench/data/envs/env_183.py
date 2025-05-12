import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 2 to 50 (inclusive)
        self.numbers = np.arange(2, 51)
        self.num_numbers = len(self.numbers)

        # Define action and observation space
        # Actions correspond to indices of the numbers array
        self.action_space = spaces.Discrete(self.num_numbers)
        # Observation space includes:
        # - Available numbers: 1 if available, 0 if not (indices 0 to 48)
        # - Last selected number, scaled between 0 and 1 (index 49)
        # - Current player: +1 or -1 (index 50)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_numbers + 2,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers are initially available
        self.available_numbers = np.ones(self.num_numbers, dtype=np.float32)
        # No number has been selected yet
        self.last_selected_number = 0.0  # Scaled between 0 and 1
        # Current player: +1 or -1
        self.current_player = 1
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}
        # Check if action is valid (number is available)
        if (
            action < 0
            or action >= self.num_numbers
            or self.available_numbers[action] == 0
        ):
            # Invalid move
            self.done = True
            reward = -10.0
            return self._get_observation(), reward, True, False, {}
        selected_number = self.numbers[action]

        # Check if move is valid according to the game rules
        if self.last_selected_number == 0.0:
            # First move can be any number
            valid_move = True
        else:
            last_number = self.last_selected_number * 50.0  # Scale back
            if selected_number % last_number == 0 or last_number % selected_number == 0:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move
            self.done = True
            reward = -10.0
            return self._get_observation(), reward, True, False, {}

        # Valid move: Update game state
        self.available_numbers[action] = 0.0
        self.last_selected_number = selected_number / 50.0  # Scale between 0 and 1

        # Check if the next player has any valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player wins
            self.done = True
            reward = +1.0
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1
        # Continue game
        reward = 0.0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Show available numbers
        available_numbers = self.numbers[self.available_numbers == 1.0]
        available_numbers_str = ", ".join(map(str, available_numbers))
        # Last selected number
        if self.last_selected_number == 0.0:
            last_number = "None"
        else:
            last_number = str(int(self.last_selected_number * 50.0))
        # Current player
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        render_str = (
            f"Available Numbers: {available_numbers_str}\n"
            f"Last Selected Number: {last_number}\n"
            f"Current Turn: {player_str}"
        )
        return render_str

    def valid_moves(self):
        valid_moves = []
        if self.done:
            return valid_moves
        if self.last_selected_number == 0.0:
            # First move: Any available number is valid
            valid_moves = [
                i for i in range(self.num_numbers) if self.available_numbers[i] == 1.0
            ]
        else:
            last_number = int(self.last_selected_number * 50.0)
            for i in range(self.num_numbers):
                if self.available_numbers[i] == 1.0:
                    number = self.numbers[i]
                    if number % last_number == 0 or last_number % number == 0:
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        # Build observation array
        observation = np.zeros(self.num_numbers + 2, dtype=np.float32)
        # Available numbers
        observation[: self.num_numbers] = self.available_numbers
        # Last selected number, scaled between 0 and 1
        observation[self.num_numbers] = self.last_selected_number
        # Current player: +1 or -1
        observation[self.num_numbers + 1] = float(self.current_player)
        return observation
