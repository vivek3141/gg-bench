import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Discrete(9) for numbers 1-9
        self.action_space = spaces.Discrete(9)

        # Define observation space:
        #   - Number pool: 9 elements (1 if available, 0 if used)
        #   - Player 1 chain: 9 elements (numbers picked by player 1, 0 for empty)
        #   - Player 2 chain: 9 elements (numbers picked by player 2, 0 for empty)
        # Thus, observation is array of length 27
        self.observation_space = spaces.Box(low=0, high=9, shape=(27,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(9, dtype=int)  # 1 indicates the number is available
        self.player1_chain = np.zeros(9, dtype=int)
        self.player2_chain = np.zeros(9, dtype=int)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        observation = self._get_observation()
        return observation, self.info  # Return observation and info

    def _get_observation(self):
        # Combine number_pool, player1_chain, player2_chain into one array
        observation = np.concatenate(
            [self.number_pool, self.player1_chain, self.player2_chain]
        )
        return observation

    def valid_moves(self, player=None):
        if player is None:
            player = self.current_player

        # Get player's chain
        if player == 1:
            chain = self.player1_chain
        else:
            chain = self.player2_chain

        # Get last number in chain (if any)
        if np.any(chain != 0):
            last_number = chain[chain != 0][-1]
            # Available numbers
            available_numbers = np.where(self.number_pool == 1)[0] + 1  # Numbers 1-9
            # Valid numbers are factors or multiples of last_number
            valid_numbers = [
                num
                for num in available_numbers
                if last_number % num == 0 or num % last_number == 0
            ]
            valid_actions = [
                num - 1 for num in valid_numbers
            ]  # Convert numbers to action indices
        else:
            # Chain is empty; any available number is valid
            valid_actions = list(np.where(self.number_pool == 1)[0])
        return valid_actions

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, self.info

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()

        if not valid_actions:
            # No valid moves; current player loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, self.info

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, self.info

        selected_number = action + 1

        # Valid move
        # Get current player's chain
        if self.current_player == 1:
            chain = self.player1_chain
        else:
            chain = self.player2_chain

        # Add selected number to chain
        chain[np.argmax(chain == 0)] = selected_number

        # Remove number from pool
        self.number_pool[action] = 0

        # Check if opponent has valid moves
        next_player = 2 if self.current_player == 1 else 1
        valid_moves_opponent = self.valid_moves(player=next_player)
        if not valid_moves_opponent:
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, self.info
        else:
            # Game continues
            # Switch to next player
            self.current_player = next_player
            reward = 0
            return self._get_observation(), reward, self.done, False, self.info

    def render(self):
        output = f"Current Player: Player {self.current_player}\n"
        output += "Number Pool: "
        pool_numbers = [str(i + 1) for i in range(9) if self.number_pool[i] == 1]
        output += ", ".join(pool_numbers) + "\n"
        output += (
            "Player 1's Chain: "
            + ", ".join(str(num) for num in self.player1_chain if num != 0)
            + "\n"
        )
        output += (
            "Player 2's Chain: "
            + ", ".join(str(num) for num in self.player2_chain if num != 0)
            + "\n"
        )
        return output
