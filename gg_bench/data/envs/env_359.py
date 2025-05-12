import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 20 possible numbers to choose from (0 to 19 representing numbers 1 to 20)
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # - First 20 entries for the shared pool (-1, 0, 1)
        # - Next 5 entries for the current player's secret pattern (numbers from 1 to 20)
        self.observation_space = spaces.Box(low=-1, high=20, shape=(25,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared pool (0: available, 1: taken by Player 1, -1: taken by Player 2)
        self.shared_pool = np.zeros(20, dtype=np.int8)

        # Randomly assign secret patterns to both players (numbers from 1 to 20)
        all_numbers = np.arange(1, 21)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.player_patterns = {}
        self.player_patterns[1] = self.np_random.choice(
            all_numbers, size=5, replace=False
        )
        self.player_patterns[-1] = self.np_random.choice(
            all_numbers, size=5, replace=False
        )

        # Initialize player collections
        self.player_collections = {1: [], -1: []}

        # Set the current player (1 or -1)
        self.current_player = 1

        # Game over flag
        self.done = False

        # Return the initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        # Check if the action is valid
        if action < 0 or action >= 20 or self.shared_pool[action] != 0 or self.done:
            # Invalid move
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        # Update the shared pool
        self.shared_pool[action] = self.current_player

        # Update the player's collection
        number_selected = action + 1  # Numbers are from 1 to 20
        self.player_collections[self.current_player].append(number_selected)

        # Check if the player has collected all numbers in their pattern
        if set(self.player_patterns[self.current_player]).issubset(
            set(self.player_collections[self.current_player])
        ):
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check if the shared pool is exhausted
        if np.all(self.shared_pool != 0):
            # Compare the number of pattern numbers collected
            current_player_count = len(
                set(self.player_patterns[self.current_player]).intersection(
                    self.player_collections[self.current_player]
                )
            )
            opponent = -self.current_player
            opponent_count = len(
                set(self.player_patterns[opponent]).intersection(
                    self.player_collections[opponent]
                )
            )

            if current_player_count > opponent_count:
                # Current player wins
                self.done = True
                return self._get_observation(), 1, True, False, {}
            elif current_player_count == opponent_count:
                # Sudden death round
                remaining_numbers = np.where(self.shared_pool == 0)[0]

                # If no numbers are left, it's a tie
                if len(remaining_numbers) == 0:
                    self.done = True
                    return self._get_observation(), 0, True, False, {}

                # Continue the game with remaining numbers
                self.done = False
            else:
                # Current player loses
                self.done = True
                return self._get_observation(), -1, True, False, {}
        else:
            self.done = False

        # Switch turns
        self.current_player *= -1

        # Continue the game
        return self._get_observation(), 0, self.done, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        output = ""
        output += "Shared Number Pool:\n"
        for i in range(20):
            owner = self.shared_pool[i]
            number = i + 1
            if owner == 0:
                output += f"{number}: Available\n"
            elif owner == 1:
                output += f"{number}: Taken by Player 1\n"
            elif owner == -1:
                output += f"{number}: Taken by Player 2\n"

        output += f"\nCurrent Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += (
            f"Your Collection: {sorted(self.player_collections[self.current_player])}\n"
        )

        return output

    def valid_moves(self):
        # Return a list of valid moves (indices of available numbers)
        return [i for i in range(20) if self.shared_pool[i] == 0]

    def _get_observation(self):
        # Compose the observation
        # First 20 entries: shared pool (-1, 0, 1)
        # Next 5 entries: current player's secret pattern (numbers from 1 to 20)
        observation = np.zeros(25, dtype=np.int8)
        observation[:20] = self.shared_pool
        observation[20:] = self.player_patterns[self.current_player]
        return observation
