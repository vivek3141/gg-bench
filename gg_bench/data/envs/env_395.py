import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 - Recruit, 1-3 - Move unit in slot 0-2
        self.action_space = spaces.Discrete(4)
        # Observation space: Battlefield positions 0-10 with unit strengths
        # Positive values for Player 1, negative for Player 2
        self.observation_space = spaces.Box(low=-5, high=5, shape=(11,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the battlefield
        self.battlefield = np.zeros(11, dtype=np.int8)
        # Initialize units for both players (max 3 units each)
        # Each unit is a dict with 'position' and 'strength'
        self.units_p1 = [None, None, None]
        self.units_p2 = [None, None, None]
        # Place starting units at bases
        self.units_p1[0] = {"position": 0, "strength": 1}
        self.units_p2[0] = {"position": 10, "strength": 1}
        self._update_battlefield()
        # Set current player (1 or 2)
        self.current_player = 1
        # Game state
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        reward = 0
        # Perform action
        if action == 0:
            # Recruit
            success = self._recruit_unit()
            if not success:
                # Should not happen due to valid_moves check
                self.done = True
                return self._get_observation(), -10, True, False, {}
        else:
            # Move unit in slot (action - 1)
            success, victory = self._move_unit(action - 1)
            if victory:
                # Current player wins
                self.done = True
                return self._get_observation(), 1, True, False, {}
            if not success:
                # Should not happen due to valid_moves check
                self.done = True
                return self._get_observation(), -10, True, False, {}

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Update observation
        observation = self._get_observation()
        return observation, reward, False, False, {}

    def render(self):
        # Return a string representation of the battlefield
        battlefield_str = ""
        for pos in range(11):
            unit = self.battlefield[pos]
            if unit == 0:
                battlefield_str += "[   ]"
            elif unit > 0:
                battlefield_str += f"[P1({unit})]"
            else:
                battlefield_str += f"[P2({-unit})]"
        return battlefield_str

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        # Check if recruit is possible
        units = self.units_p1 if self.current_player == 1 else self.units_p2
        if any(u is None for u in units):
            valid_actions.append(0)  # Recruit action
        # Check for movable units
        for idx, unit in enumerate(units):
            if unit is not None:
                valid_actions.append(idx + 1)  # Move actions
        return valid_actions

    def _recruit_unit(self):
        # Recruit a new unit at the base if possible
        units = self.units_p1 if self.current_player == 1 else self.units_p2
        base_position = 0 if self.current_player == 1 else 10
        for idx in range(3):
            if units[idx] is None:
                # Place new unit in this slot
                units[idx] = {"position": base_position, "strength": 1}
                self._update_battlefield()
                return True
        # Cannot recruit (no free slot)
        return False

    def _move_unit(self, slot_idx):
        # Move the unit in the given slot
        units = self.units_p1 if self.current_player == 1 else self.units_p2
        enemy_units = self.units_p2 if self.current_player == 1 else self.units_p1
        if slot_idx < 0 or slot_idx > 2 or units[slot_idx] is None:
            return False, False  # Invalid move

        unit = units[slot_idx]
        # Increase strength by 1 (max 5)
        unit["strength"] = min(unit["strength"] + 1, 5)
        # Move unit towards opponent's base
        if self.current_player == 1:
            new_position = unit["position"] + 1
        else:
            new_position = unit["position"] - 1

        if new_position < 0 or new_position > 10:
            return False, False  # Invalid move (should not happen)

        # Check for battle
        battlefield_unit = self.battlefield[new_position]
        if battlefield_unit == 0:
            # No enemy unit, move is successful
            unit["position"] = new_position
        else:
            # Battle occurs
            enemy_unit_strength = abs(battlefield_unit)
            if self.current_player == 1 and battlefield_unit < 0:
                # Enemy unit is Player 2
                result = self._resolve_battle(
                    unit, units, slot_idx, enemy_units, new_position
                )
            elif self.current_player == 2 and battlefield_unit > 0:
                # Enemy unit is Player 1
                result = self._resolve_battle(
                    unit, units, slot_idx, enemy_units, new_position
                )
            else:
                # Should not happen (no friendly fire)
                return False, False

            if result == "victory":
                # Moved into opponent's base
                return True, True
            elif result == "continue":
                pass  # Battle resolved, continue game
            else:
                return False, False  # Invalid move (should not happen)

        # Check for victory (moving into opponent's base)
        if (self.current_player == 1 and unit["position"] == 10) or (
            self.current_player == 2 and unit["position"] == 0
        ):
            # Victory
            return True, True

        self._update_battlefield()
        return True, False

    def _resolve_battle(self, unit, units, slot_idx, enemy_units, position):
        # Find the enemy unit at the position
        for idx, enemy_unit in enumerate(enemy_units):
            if enemy_unit is not None and enemy_unit["position"] == position:
                break
        else:
            # Enemy unit not found (should not happen)
            return "error"

        # Battle resolution
        if unit["strength"] > enemy_unit["strength"]:
            # Player wins, enemy unit is removed
            enemy_units[idx] = None
            unit["position"] = position
        elif unit["strength"] < enemy_unit["strength"]:
            # Player's unit is removed
            units[slot_idx] = None
        else:
            # Both units are removed
            units[slot_idx] = None
            enemy_units[idx] = None

        self._update_battlefield()
        return "continue"

    def _update_battlefield(self):
        # Update the battlefield array based on units' positions
        self.battlefield[:] = 0
        for unit in self.units_p1:
            if unit is not None:
                self.battlefield[unit["position"]] = unit["strength"]
        for unit in self.units_p2:
            if unit is not None:
                self.battlefield[unit["position"]] = -unit["strength"]

    def _get_observation(self):
        # Return a copy of the battlefield as the observation
        return self.battlefield.copy()
