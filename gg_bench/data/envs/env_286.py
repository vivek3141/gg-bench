import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 5 spells, actions are 0 to 4
        self.action_space = spaces.Discrete(5)

        # Observation space consists of:
        # [current_player_HP, opponent_HP, current_player_shield, opponent_shield]
        # HPs range from 0 to 10
        # Shields are 0 (inactive) or 1 (active)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([10, 10, 1, 1]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Health Points for both players
        self.HPs = np.array(
            [10, 10], dtype=np.int32
        )  # Index 0: Player A, Index 1: Player B

        # Shield status for both players (0: Inactive, 1: Active)
        self.shields = np.array([0, 0], dtype=np.int32)

        # Set current player (0: Player A, 1: Player B)
        self.current_player = 0

        # Game over flag
        self.done = False

        # Return the initial observation and info
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation from the perspective of the current player
        obs = np.array(
            [
                self.HPs[self.current_player],
                self.HPs[1 - self.current_player],
                self.shields[self.current_player],
                self.shields[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        current = self.current_player
        opponent = 1 - current
        reward = 0
        info = {}

        # Spells:
        # 0: Firebolt - Deals 1 damage to opponent
        # 1: Lightning Strike - Deals 2 damage to opponent
        # 2: Dark Pact - Deals 3 damage to opponent, costs 1 HP from caster
        # 3: Shield - Activates a shield
        # 4: Healing Touch - Restores 2 HP to caster (max 10 HP)

        # Handle self-effects first
        if action == 2:  # Dark Pact
            self.HPs[current] -= 1  # Caster loses 1 HP
        elif action == 4:  # Healing Touch
            self.HPs[current] = min(self.HPs[current] + 2, 10)  # Heal up to max 10 HP

        # Check for self-defeat
        if self.HPs[current] <= 0:
            self.done = True
            reward = -1  # Current player loses
            return self._get_obs(), reward, True, False, info

        # Handle spells that affect the opponent
        if action == 0:  # Firebolt
            damage = 1
        elif action == 1:  # Lightning Strike
            damage = 2
        elif action == 2:  # Dark Pact
            damage = 3
        else:
            damage = 0

        if damage > 0:
            if self.shields[opponent]:
                self.shields[opponent] = (
                    0  # Shield blocks the damage and then deactivates
                )
            else:
                self.HPs[opponent] -= damage  # Apply damage to opponent

        # Handle Shield activation
        if action == 3:  # Shield
            # Activate shield if not already active
            self.shields[current] = 1

        # Check for opponent defeat
        if self.HPs[opponent] <= 0:
            self.done = True
            reward = 1  # Current player wins
            return self._get_obs(), reward, True, False, info

        # Switch turns to the opponent
        self.current_player = opponent

        # Continue the game
        return self._get_obs(), reward, False, False, info

    def render(self):
        # Provide a string representation of the game state
        state = "--- Battle Wizards Duel ---\n"
        state += f"Player {self.current_player + 1}'s Turn\n"
        for i in range(2):
            player = "Player A" if i == 0 else "Player B"
            hp = self.HPs[i]
            shield_status = "Active" if self.shields[i] else "Inactive"
            state += f"{player} - HP: {hp}, Shield: {shield_status}\n"
        return state

    def valid_moves(self):
        # All spells are valid as long as the game is not over
        if self.done:
            return []
        return [0, 1, 2, 3, 4]
