import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the action and observation space
        # Actions: 0 = Charge, 1 = Attack
        self.action_space = spaces.Discrete(2)
        self.max_attack_power = 100  # Set maximum possible Attack Power
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 1, 0, 1], dtype=np.int32
            ),  # [player_power, player_attack, opponent_power, opponent_attack]
            high=np.array(
                [10, self.max_attack_power, 10, self.max_attack_power], dtype=np.int32
            ),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.power_levels = [10, 10]  # [player 0 power, player 1 power]
        self.attack_powers = [1, 1]  # [player 0 attack power, player 1 attack power]
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_obs(), -10, True, False, {}
        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Current player action
        opponent = 1 - self.current_player

        if action == 0:  # Charge
            self.attack_powers[self.current_player] += 1
            # Ensure Attack Power does not exceed max_attack_power
            self.attack_powers[self.current_player] = min(
                self.attack_powers[self.current_player], self.max_attack_power
            )
        elif action == 1:  # Attack
            # Deal damage equal to current player's Attack Power
            damage = self.attack_powers[self.current_player]
            self.power_levels[opponent] -= damage
            # Reset current player's Attack Power to 1
            self.attack_powers[self.current_player] = 1
        else:
            # Invalid action (should not reach here due to action_space check)
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check for win condition
        if self.power_levels[opponent] <= 0:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch current player
        self.current_player = opponent

        return self._get_obs(), 0, False, False, {}

    def render(self):
        opponent = 1 - self.current_player
        return (
            f"--- Current Player: Player {self.current_player + 1} ---\n"
            f"Your Power Level: {self.power_levels[self.current_player]} | "
            f"Your Attack Power: {self.attack_powers[self.current_player]}\n"
            f"Opponent's Power Level: {self.power_levels[opponent]} | "
            f"Opponent's Attack Power: {self.attack_powers[opponent]}"
        )

    def valid_moves(self):
        # Both actions are always valid in this game
        return [0, 1]

    def _get_obs(self):
        opponent = 1 - self.current_player
        return np.array(
            [
                self.power_levels[self.current_player],
                self.attack_powers[self.current_player],
                self.power_levels[opponent],
                self.attack_powers[opponent],
            ],
            dtype=np.int32,
        )
