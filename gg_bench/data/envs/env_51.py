import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers from 1 to 9 (represented as actions 0 to 8)
        self.action_space = spaces.Discrete(9)
        # Observation space: both players' towers (6 levels), values from 0 (empty) to 9
        self.observation_space = spaces.Box(low=0, high=9, shape=(6,), dtype=np.int32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the towers: 2 players, 3 levels each
        self.player_towers = np.zeros((2, 3), dtype=np.int32)
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number_selected = action + 1  # Map action to number (0->1, ..., 8->9)

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move: penalize and forfeit turn
            reward = -10
            # Forfeit turn by switching player
            self.current_player = 1 - self.current_player

            # Check if the next player can make a move
            if len(self.valid_moves()) == 0:
                # Next player cannot move
                # Switch back to check if current player can move
                self.current_player = 1 - self.current_player
                if len(self.valid_moves()) == 0:
                    # Neither player can move; game over
                    self.done = True
                    return self._get_observation(), reward, True, False, {}
                else:
                    # Current player plays again
                    return self._get_observation(), reward, False, False, {}
            else:
                # Game continues with next player
                return self._get_observation(), reward, False, False, {}
        else:
            # Valid move: place the number
            current_tower = self.player_towers[self.current_player]
            empty_levels = np.where(current_tower == 0)[0]
            level = empty_levels[0]  # Lowest available level
            current_tower[level] = number_selected

            if level == 2:
                # Player has completed their tower; they win
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, {}
            else:
                # Continue game
                reward = 0
                # Switch player
                self.current_player = 1 - self.current_player

                # Check if the next player can make a move
                if len(self.valid_moves()) == 0:
                    # Next player cannot move
                    # Switch back to check if current player can move
                    self.current_player = 1 - self.current_player
                    if len(self.valid_moves()) == 0:
                        # Neither player can move; game over
                        self.done = True
                        return self._get_observation(), reward, True, False, {}
                    else:
                        # Current player plays again
                        return self._get_observation(), reward, False, False, {}
                else:
                    # Game continues with next player
                    return self._get_observation(), reward, False, False, {}

    def render(self):
        output = ""
        for player_index in [0, 1]:
            tower = self.player_towers[player_index]
            output += f"Player {player_index + 1}'s Tower:\n"
            for level in range(3):
                output += f"  Level {level + 1}: {tower[level]}\n"
            output += "\n"
        output += f"Current Player: Player {self.current_player + 1}\n"
        return output

    def valid_moves(self):
        # Return valid actions (integers from 0 to 8)
        valid_numbers = self.get_valid_numbers_for_player(self.current_player)
        valid_actions = [n - 1 for n in valid_numbers]  # Map numbers to actions
        return valid_actions

    def get_valid_numbers_for_player(self, player_index):
        tower = self.player_towers[player_index]
        empty_levels = np.where(tower == 0)[0]
        if len(empty_levels) == 0:
            # Tower is full; no valid moves
            return []
        level = empty_levels[0]
        if level == 0:
            # Level 1: any number from 1 to 9 is valid
            valid_numbers = list(range(1, 10))
        else:
            previous_number = tower[level - 1]
            # Numbers less than the previous level's number
            valid_numbers = [n for n in range(1, 10) if n < previous_number]
        return valid_numbers

    def _get_observation(self):
        # Combine both players' towers into a single observation array
        observation = np.concatenate((self.player_towers[0], self.player_towers[1]))
        return observation
