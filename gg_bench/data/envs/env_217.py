import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete space of 30 actions representing numbers 1 to 30
        self.action_space = spaces.Discrete(30)

        # Observation space: 31-element array
        # First 30 elements represent availability of numbers 1 to 30 (1: available, 0: selected)
        # The 31st element represents the last number selected by the opponent (0 if none)
        self.observation_space = spaces.Box(low=0, high=30, shape=(31,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 1 to 30 are initially available (1 indicates available)
        self.available_numbers = np.ones(30, dtype=np.int32)
        # Last number selected by the opponent; 0 indicates none
        self.last_opponent_move = 0
        # Current player indicator (1 or -1); not necessary for self-play but kept for clarity
        self.current_player = 1
        self.done = False
        # Return initial observation and empty info
        return self._get_observation(), {}

    def step(self, action):
        # Map action to number (actions are 0-29 corresponding to numbers 1-30)
        number = action + 1

        # Check if action is within valid range
        if action < 0 or action >= 30:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Check if the number is available
        if self.available_numbers[action] == 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Validate the move according to game rules
        valid = False
        # First move
        if self.last_opponent_move == 0:
            # Cannot select 1 on the first move
            if number != 1:
                valid = True
            else:
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}
        else:
            # Subsequent moves
            # Check if number is a divisor or multiple of the opponent's last move
            if number != 1 and (
                self.last_opponent_move % number == 0
                or number % self.last_opponent_move == 0
            ):
                valid = True
            elif number == 1:
                # Check if there are no other valid moves besides selecting 1
                has_other_valid_moves = False
                for i in range(30):
                    num = i + 1
                    if (
                        self.available_numbers[i] == 1
                        and num != 1
                        and (
                            self.last_opponent_move % num == 0
                            or num % self.last_opponent_move == 0
                        )
                    ):
                        has_other_valid_moves = True
                        break
                if not has_other_valid_moves:
                    valid = True  # Selecting 1 is valid if no other valid moves
                else:
                    reward = -10
                    self.done = True
                    return self._get_observation(), reward, self.done, False, {}
            else:
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, False, {}

        if valid:
            # Update the game state
            self.available_numbers[action] = 0  # Mark the number as selected
            self.last_opponent_move = number

            # Check if the next player has any valid moves
            opponent_valid_moves = self._get_valid_moves()
            if not opponent_valid_moves:
                # Current player wins
                reward = 1
                self.done = True
                return self._get_observation(), reward, self.done, False, {}
            else:
                # Switch to the next player (for completeness; not needed in self-play)
                self.current_player *= -1
                reward = -10  # Penalty per valid move to encourage quick winning
                return self._get_observation(), reward, self.done, False, {}
        else:
            # Should not reach here due to earlier checks
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # Generate a string representation of the game state
        available_nums = [
            str(i + 1) for i in range(30) if self.available_numbers[i] == 1
        ]
        selected_nums = [
            str(i + 1) for i in range(30) if self.available_numbers[i] == 0
        ]
        state_str = "Available Numbers: " + ", ".join(available_nums) + "\n"
        state_str += "Selected Numbers: " + ", ".join(selected_nums) + "\n"
        state_str += f"Last Opponent's Move: {self.last_opponent_move}\n"
        return state_str

    def valid_moves(self):
        # Return the list of valid actions (indices) for the current player
        return self._get_valid_moves()

    def _get_observation(self):
        # Construct the observation array
        # First 30 elements are availability of numbers 1-30
        # 31st element is the last opponent's move
        observation = np.append(self.available_numbers.copy(), self.last_opponent_move)
        return observation

    def _get_valid_moves(self):
        valid_moves = []
        # If this is the first move (no last opponent move), valid moves are any available numbers 2-30
        if self.last_opponent_move == 0:
            for i in range(1, 30):  # Numbers 2-30
                if self.available_numbers[i] == 1:
                    valid_moves.append(i)
        else:
            for i in range(30):
                number = i + 1
                if self.available_numbers[i] == 1:
                    if number != 1 and (
                        self.last_opponent_move % number == 0
                        or number % self.last_opponent_move == 0
                    ):
                        valid_moves.append(i)
            # If no valid moves found, check if selecting 1 is valid
            if not valid_moves and self.available_numbers[0] == 1:
                # Check if selecting 1 is valid (only if no other valid moves)
                valid_moves.append(0)  # Action corresponding to number 1
        return valid_moves
