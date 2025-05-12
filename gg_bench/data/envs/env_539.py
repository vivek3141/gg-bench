import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(100), actions are indices from 0 to 99 corresponding to numbers 1 to 100
        self.action_space = spaces.Discrete(100)

        # Observation space: Box(low=-1, high=100, shape=(102,), dtype=np.int8)
        # Observation includes:
        # observation[0]: Current player (1 for Player 1, -1 for Player 2)
        # observation[1]: Last number placed on the stack (0 if stack is empty)
        # observation[2:]: Used numbers, binary (0 or 1), length 100
        self.observation_space = spaces.Box(
            low=-1, high=100, shape=(102,), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stack = []
        self.used_numbers = np.zeros(100, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        self.winner = None

        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number = action + 1  # Map action index to the number 1 to 100

        # Check if number has been used before
        if self.used_numbers[number - 1] == 1:
            # Invalid move: number has been used
            self.done = True
            self.winner = -self.current_player  # The other player wins
            reward = -10  # For current player
            return self._get_observation(), reward, True, False, {}

        # Check if the move is valid according to the game rules
        valid = False

        if len(self.stack) == 0:
            # First move: any number from 1 to 10
            if 1 <= number <= 10:
                valid = True
        else:
            top_number = self.stack[-1]
            # Check if number is a multiple or factor of top_number
            if number % top_number == 0 or top_number % number == 0:
                valid = True

        if not valid:
            # Invalid move
            self.done = True
            self.winner = -self.current_player  # The other player wins
            reward = -10  # For current player
            return self._get_observation(), reward, True, False, {}

        # Valid move: update the state
        self.stack.append(number)
        self.used_numbers[number - 1] = 1

        # Check if next player has any valid moves
        next_player = -self.current_player

        # Switch current_player temporarily to next_player to compute observation correctly
        self.current_player = next_player
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Next player cannot make a move
            self.done = True
            self.winner = -self.current_player  # Previous player wins
            # Switch back to previous player to return correct observation
            self.current_player = -self.current_player
            reward = 1  # For current player (who just made the move)
            return self._get_observation(), reward, True, False, {}
        else:
            # Game continues
            # current_player remains as next player
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        state_str = "Current Player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        state_str += "Stack: {}\n".format(self.stack)
        return state_str

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = []

        if len(self.stack) == 0:
            # First move: numbers from 1 to 10 that have not been used
            for i in range(10):
                if self.used_numbers[i] == 0:
                    valid_actions.append(i)  # action index corresponds to number i+1
        else:
            top_number = self.stack[-1]
            for i in range(100):
                number = i + 1
                if self.used_numbers[i] == 1:
                    continue
                if number % top_number == 0 or top_number % number == 0:
                    valid_actions.append(i)
        return valid_actions

    def _get_observation(self):
        observation = np.zeros(102, dtype=np.int8)
        observation[0] = self.current_player
        if len(self.stack) == 0:
            observation[1] = 0
        else:
            observation[1] = self.stack[-1]
        observation[2:] = self.used_numbers
        return observation
