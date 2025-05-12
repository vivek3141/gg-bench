import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 7 planets and 7 ships, total possible actions = 7 * 7 = 49
        self.action_space = spaces.Discrete(49)

        # Observation space: planets (7), own ships (7), opponent ships (7), current player (1)
        # Planets: 0 (unconquered), 1 (conquered by current player), -1 (conquered by opponent)
        # Ships: 1 (available), 0 (expended)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(22,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.planets = np.zeros(
            7, dtype=np.int8
        )  # 0: unconquered, 1: Player 1, -1: Player 2
        self.ships = {
            1: np.ones(7, dtype=np.int8),  # Player 1 ships: 1 (available), 0 (expended)
            -1: np.ones(7, dtype=np.int8),  # Player 2 ships
        }
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action index to planet_number and ship_number
        planet_number = action // 7 + 1  # Planet numbers from 1 to 7
        ship_number = action % 7 + 1  # Ship numbers from 1 to 7
        planet_idx = planet_number - 1  # Zero-based index
        ship_idx = ship_number - 1  # Zero-based index

        # Check if the planet is unconquered
        if self.planets[planet_idx] != 0:
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Check if the current player's ship is available
        if self.ships[self.current_player][ship_idx] != 1:
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Ship is expended regardless of outcome
        self.ships[self.current_player][ship_idx] = 0

        # Check if attack is successful
        ship_power = ship_number
        planet_defense = planet_number
        if ship_power >= planet_defense:
            # Attack successful, conquer the planet
            self.planets[planet_idx] = self.current_player
        # If attack fails, planet remains unconquered

        # Check victory condition
        planets_conquered = np.sum(self.planets == self.current_player)
        if planets_conquered >= 4:
            reward = 1
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        # Observation vector: planets (7), own ships (7), opponent ships (7), current player (1)
        planets_obs = np.zeros(7, dtype=np.int8)
        for idx in range(7):
            if self.planets[idx] == 0:
                planets_obs[idx] = 0
            elif self.planets[idx] == self.current_player:
                planets_obs[idx] = 1
            else:
                planets_obs[idx] = -1

        own_ships_obs = self.ships[self.current_player]
        opponent_ships_obs = self.ships[-self.current_player]
        current_player_obs = np.array([self.current_player], dtype=np.int8)
        observation = np.concatenate(
            [planets_obs, own_ships_obs, opponent_ships_obs, current_player_obs]
        )
        return observation

    def render(self):
        # Return a visual representation of the game state as a string
        s = "Planets:\n"
        for idx in range(7):
            status = self.planets[idx]
            if status == 0:
                owner = "Unconquered"
            elif status == 1:
                owner = "Conquered by Player 1"
            else:
                owner = "Conquered by Player 2"
            s += f"[{idx+1}] {owner}  "
        s += "\n\n"

        s += f"Current Player: Player {1 if self.current_player == 1 else 2}\n\n"

        s += "Ships:\n"
        s += "Player 1 Ships Available: "
        s += ", ".join(
            str(i + 1) for i, available in enumerate(self.ships[1]) if available == 1
        )
        s += "\n"
        s += "Player 2 Ships Available: "
        s += ", ".join(
            str(i + 1) for i, available in enumerate(self.ships[-1]) if available == 1
        )
        s += "\n"

        return s

    def valid_moves(self):
        valid_actions = []
        for planet_idx in range(7):
            if self.planets[planet_idx] != 0:
                continue  # Skip conquered planets
            for ship_idx in range(7):
                if self.ships[self.current_player][ship_idx] == 1:
                    # Valid action
                    action = planet_idx * 7 + ship_idx
                    valid_actions.append(action)
        return valid_actions
