import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 for 'charge', 1 for 'attack'
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation is an array: [my_power_core, my_charge_level, opponent_power_core, opponent_charge_level]
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, 1]), high=np.array([10, 5, 10, 5]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_power_cores = [10, 10]  # Player 0 and Player 1
        self.player_charge_levels = [1, 1]  # Charge Levels for both players
        self.current_player = 0  # Player 0 starts
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Observation from current player's perspective
        obs = np.array(
            [
                self.player_power_cores[self.current_player],
                self.player_charge_levels[self.current_player],
                self.player_power_cores[1 - self.current_player],
                self.player_charge_levels[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        terminated = False

        # Check for valid action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            reward = -10
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        reward = 0

        if action == 0:  # 'charge'
            # Increase current player's charge level by 1, up to a max of 5
            self.player_charge_levels[self.current_player] += 1
            if self.player_charge_levels[self.current_player] > 5:
                self.player_charge_levels[self.current_player] = 5

        elif action == 1:  # 'attack'
            # Deal damage equal to current charge level to the opponent
            damage = self.player_charge_levels[self.current_player]
            opponent = 1 - self.current_player
            self.player_power_cores[opponent] -= damage
            # Reset current player's charge level to 1
            self.player_charge_levels[self.current_player] = 1
            # Check for win condition
            if self.player_power_cores[opponent] <= 0:
                reward = 1  # Current player wins
                terminated = True
                return self._get_obs(), reward, terminated, False, {}

        # Switch to other player if game not over
        if not terminated:
            self.current_player = 1 - self.current_player

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        # Return a string representing the current game state
        s = "---------------------------\n"
        s += f"Player {self.current_player + 1}'s Turn\n"
        s += "---------------------------\n"
        s += f"Your Power Core: {self.player_power_cores[self.current_player]}\n"
        s += f"Your Charge Level: {self.player_charge_levels[self.current_player]}\n"
        s += f"Opponent's Power Core: {self.player_power_cores[1 - self.current_player]}\n"
        s += f"Opponent's Charge Level: {self.player_charge_levels[1 - self.current_player]}\n"
        return s

    def valid_moves(self):
        # Return a list of valid actions for the current player
        # Actions: 0 - 'charge', 1 - 'attack'
        actions = [1]  # 'attack' is always valid
        if self.player_charge_levels[self.current_player] < 5:
            actions.append(0)  # 'charge' is valid if Charge Level < 5
        return actions
