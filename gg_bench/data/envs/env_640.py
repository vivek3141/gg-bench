import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers 1-20 (actions 0-19), pass (action 20)
        self.action_space = spaces.Discrete(21)
        # Observation space:
        # - Shared pool: 20 numbers (1 if available, 0 if taken)
        # - Player 1 collection: 20 numbers (1 if the number is in the collection)
        # - Player 2 collection: 20 numbers
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(60,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared number pool with numbers 1 to 20
        self.shared_pool = np.ones(20, dtype=np.float32)
        # Players' collections
        self.player_collections = [
            np.zeros(20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
        ]
        # Players' total sums
        self.player_sums = [0, 0]
        # Current player (0 or 1)
        self.current_player = 0
        # Consecutive passes by players
        self.passes = [False, False]
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        reward = 0
        info = {}

        if self.done:
            return self._get_observation(), reward, True, False, info

        if action < 0 or action > 20:
            # Invalid action index
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, info

        # Check if action is 'pass'
        if action == 20:
            # Check if player has valid moves
            if self._has_valid_moves(self.current_player):
                # Passing when valid moves are available is invalid
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, info
            else:
                # Valid pass
                self.passes[self.current_player] = True
                # Check for consecutive passes
                if all(self.passes):
                    # Game over, determine winner
                    self.done = True
                    player_sum = self.player_sums[self.current_player]
                    opponent_sum = self.player_sums[1 - self.current_player]
                    if abs(30 - player_sum) < abs(30 - opponent_sum):
                        # Current player wins
                        reward = 1
                    elif abs(30 - player_sum) > abs(30 - opponent_sum):
                        # Current player loses
                        reward = -1
                    else:
                        # Tie (unlikely), no reward
                        reward = 0
                    return self._get_observation(), reward, True, False, info
                else:
                    # Switch to the next player
                    self.current_player = 1 - self.current_player
                    return self._get_observation(), reward, False, False, info
        else:
            number = action + 1  # Numbers are from 1 to 20
            # Check if number is available in the pool
            if self.shared_pool[number - 1] == 0:
                # Number not available, invalid move
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, info
            # Check if adding the number exceeds total sum of 30
            if self.player_sums[self.current_player] + number > 30:
                # Cannot select this number, invalid move
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, info
            # Valid move
            self.shared_pool[number - 1] = 0  # Remove number from pool
            self.player_collections[self.current_player][
                number - 1
            ] = 1  # Add number to player's collection
            self.player_sums[self.current_player] += number
            # Reset pass for current player
            self.passes[self.current_player] = False
            # Check for win condition
            if self.player_sums[self.current_player] == 30:
                # Current player wins
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, info
            elif self.player_sums[self.current_player] > 30:
                # Should not happen due to earlier check, but just in case
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, info
            else:
                # Valid move, continue game
                # Switch to the next player
                self.current_player = 1 - self.current_player
                return self._get_observation(), reward, False, False, info

    def render(self):
        output = ""
        output += "Shared Pool:\n"
        pool_numbers = [str(i + 1) for i in range(20) if self.shared_pool[i] == 1]
        output += ", ".join(pool_numbers) + "\n"
        output += f"Player 1's collection (sum={int(self.player_sums[0])}):\n"
        p1_numbers = [
            str(i + 1) for i in range(20) if self.player_collections[0][i] == 1
        ]
        output += ", ".join(p1_numbers) + "\n"
        output += f"Player 2's collection (sum={int(self.player_sums[1])}):\n"
        p2_numbers = [
            str(i + 1) for i in range(20) if self.player_collections[1][i] == 1
        ]
        output += ", ".join(p2_numbers) + "\n"
        output += f"Current Player: Player {self.current_player + 1}\n"
        return output

    def valid_moves(self):
        valid_actions = []
        if self._has_valid_moves(self.current_player):
            for i in range(20):
                if (
                    self.shared_pool[i] == 1
                    and self.player_sums[self.current_player] + (i + 1) <= 30
                ):
                    valid_actions.append(i)
        else:
            # If no valid moves, the only valid action is 'pass'
            valid_actions.append(20)
        return valid_actions

    def _has_valid_moves(self, player):
        for i in range(20):
            if self.shared_pool[i] == 1 and self.player_sums[player] + (i + 1) <= 30:
                return True
        return False

    def _get_observation(self):
        observation = np.concatenate(
            (self.shared_pool, self.player_collections[0], self.player_collections[1])
        )
        return observation.astype(np.float32)
