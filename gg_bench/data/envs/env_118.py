import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 18 possible actions
        # Actions 0-8: Attack with numbers 1-9
        # Actions 9-17: Defend with numbers 1-9
        self.action_space = spaces.Discrete(18)

        # Observation space:
        # [My_HP, Opponent_HP, Available_Numbers[1-9], My_Defense_Value, Opponent_Defense_Value]
        # Total length = 2 (HPs) + 9 (Numbers) + 2 (Defense Values) = 13
        low = np.array([0, 0] + [0] * 9 + [0, 0])
        high = np.array([20, 20] + [1] * 9 + [9, 9])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_HP = [20, 20]
        self.defense_values = [0, 0]
        self.available_numbers = [1] * 9  # Numbers 1-9 are available
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation from the perspective of the current player
        my_HP = self.player_HP[self.current_player]
        opp_HP = self.player_HP[1 - self.current_player]
        my_defense = self.defense_values[self.current_player]
        opp_defense = self.defense_values[1 - self.current_player]

        obs = np.array(
            [my_HP, opp_HP] + self.available_numbers + [my_defense, opp_defense],
            dtype=np.int32,
        )
        return obs

    def valid_moves(self):
        valid_actions = []
        for action in range(18):
            if action < 9:
                # Attack with number (action +1)
                number = action + 1
            else:
                # Defend with number (action - 9 + 1)
                number = action - 9 + 1
            if self.available_numbers[number - 1]:
                valid_actions.append(action)
        return valid_actions

    def step(self, action):
        if action not in self.valid_moves() or self.done:
            # Invalid move
            return self._get_obs(), -10, True, False, {}

        # At the start of the player's turn, reset their own defense value
        self.defense_values[self.current_player] = 0

        # Process action
        if action < 9:
            # Attack
            attack_number = action + 1
            # Remove number from available numbers
            self.available_numbers[attack_number - 1] = 0

            # Calculate net damage
            opp_defense = self.defense_values[1 - self.current_player]
            net_damage = max(attack_number - opp_defense, 0)
            # Subtract net damage from opponent's HP
            self.player_HP[1 - self.current_player] -= net_damage
            # Opponent's shield reduces damage from the next attack only
            self.defense_values[1 - self.current_player] = 0
        else:
            # Defend
            defense_number = action - 9 + 1
            # Remove number from available numbers
            self.available_numbers[defense_number - 1] = 0
            # Set current player's defense value
            self.defense_values[self.current_player] = defense_number

        # Check for win
        if self.player_HP[1 - self.current_player] <= 0:
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Check if no numbers are left
        if sum(self.available_numbers) == 0:
            self.done = True
            if (
                self.player_HP[self.current_player]
                > self.player_HP[1 - self.current_player]
            ):
                # Current player wins
                return self._get_obs(), 1, True, False, {}
            elif (
                self.player_HP[self.current_player]
                < self.player_HP[1 - self.current_player]
            ):
                # Current player loses
                return self._get_obs(), -1, True, False, {}
            else:
                # No draws allowed, current player loses
                return self._get_obs(), -1, True, False, {}

        # Switch current player
        self.current_player = 1 - self.current_player

        return self._get_obs(), 0, False, False, {}

    def render(self):
        s = f"Player {self.current_player + 1}'s Turn\n"
        s += f"Your HP: {self.player_HP[self.current_player]}\n"
        s += f"Opponent's HP: {self.player_HP[1 - self.current_player]}\n"
        available_nums = [i + 1 for i, val in enumerate(self.available_numbers) if val]
        s += f"Available Numbers: {available_nums}\n"
        s += f"Your Defense Value: {self.defense_values[self.current_player]}\n"
        s += f"Opponent's Defense Value: {self.defense_values[1 - self.current_player]}\n"
        return s
