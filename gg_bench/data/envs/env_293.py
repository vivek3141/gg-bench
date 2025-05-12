import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=20):
        super(CustomEnv, self).__init__()

        self.N = N  # Maximum number in the number pool
        self.action_space = spaces.Discrete(self.N)
        # Observation space includes the number pool status and last opponent number
        # Number pool status: 0 (unclaimed), 1 (claimed by Player 1), 2 (claimed by Player 2)
        # Last opponent number: integer from 0 (no last number) to N
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.N + 1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.zeros(self.N, dtype=np.int32)  # Status of numbers 1 to N
        self.current_player = 1  # Player 1 starts the game
        self.last_opponent_number = 0  # No last opponent number at the start
        self.done = False  # Game is not over
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number_selected = action + 1  # Map action index to number in the pool (1 to N)

        # Check if selected number is unclaimed
        if self.number_pool[action] != 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Validate the move according to game rules
        if self.current_player == 1 and self.last_opponent_number == 0:
            # First move by Player 1; cannot select 1
            if number_selected == 1:
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}
        else:
            # Subsequent moves; must be factor or multiple of last opponent number
            if not (
                number_selected % self.last_opponent_number == 0
                or self.last_opponent_number % number_selected == 0
            ):
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}

        # Valid move; update game state
        self.number_pool[action] = self.current_player  # Mark number as claimed
        last_player_number = (
            number_selected  # Update last number selected by current player
        )

        # Check for win condition: selecting 1
        if number_selected == 1:
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Switch players
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        self.last_opponent_number = last_player_number  # Update last opponent number

        # Check if opponent has any valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Opponent cannot make a valid move; current player wins
            self.current_player = 3 - self.current_player  # Switch back to winner
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Continue the game
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        output = "Number Pool Status:\n"
        for i in range(self.N):
            status = self.number_pool[i]
            if status == 0:
                output += f"{i+1}: Unclaimed\n"
            elif status == 1:
                output += f"{i+1}: Claimed by Player 1\n"
            elif status == 2:
                output += f"{i+1}: Claimed by Player 2\n"
        output += f"Current Player: Player {self.current_player}\n"
        if self.last_opponent_number > 0:
            output += f"Last Opponent Number: {self.last_opponent_number}\n"
        else:
            output += "No Last Opponent Number (First Move)\n"
        return output

    def valid_moves(self):
        valid_moves = []
        for i in range(self.N):
            if self.number_pool[i] == 0:
                number = i + 1
                if self.current_player == 1 and self.last_opponent_number == 0:
                    # First move by Player 1; cannot select 1
                    if number != 1:
                        valid_moves.append(i)
                else:
                    # Subsequent moves; must be factor or multiple of last opponent number
                    if (
                        number % self.last_opponent_number == 0
                        or self.last_opponent_number % number == 0
                    ):
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        # Observation includes number pool status and last opponent number
        observation = np.append(self.number_pool, self.last_opponent_number)
        return observation
