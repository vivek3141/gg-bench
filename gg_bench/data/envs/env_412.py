import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete actions from 0 to 10
        # Action 0 to 10 corresponds to selecting a number or block amount
        self.action_space = spaces.Discrete(11)

        # Observation space:
        # Index 0: Current player's HP (scaled between 0 and 1)
        # Index 1: Opponent's HP (scaled between 0 and 1)
        # Indices 2-11: Number Pool availability for numbers 1-10 (1 if available, 0 if used)
        # Index 12: Phase indicator (0 for attack phase, 1 for defense phase)
        # Index 13: Attack number (scaled between 0 and 1), 0 if not in defense phase
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [50, 50]  # HP for both players: [current_player, opponent]
        self.number_pool = np.ones(10, dtype=np.float32)  # Numbers 1-10 available
        self.current_player = 0  # Player 0 starts
        self.phase = "attack"  # 'attack' or 'defense'
        self.attack_number = 0  # The number used in the attack
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}

        reward = -10  # Default reward for a valid move
        info = {}

        if self.phase == "attack":
            valid_actions = self.valid_moves()
            if action not in valid_actions:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            # Attack phase: select a number from the Number Pool
            self.attack_number = action  # Attack number for defense phase
            self.number_pool[action - 1] = 0  # Remove the number from the pool
            self.phase = "defense"  # Switch to defense phase
            # No HP changes in attack phase
        elif self.phase == "defense":
            # Defense phase: choose block amount between 0 and attack_number
            if action < 0 or action > self.attack_number:
                self.done = True
                return self._get_obs(), -10, True, False, {}
            block_amount = action
            damage_taken = self.attack_number - block_amount
            total_hp_lost = block_amount + damage_taken
            # Reduce defender's HP
            defender = self.current_player
            self.player_hp[defender] -= total_hp_lost
            # Check for defeat
            if self.player_hp[defender] <= 0:
                reward = 1  # Current player wins
                self.done = True
                return self._get_obs(), reward, True, False, {}
            else:
                # Switch roles and phase
                self.current_player = 1 - self.current_player
                self.phase = "attack"
                self.attack_number = 0  # Reset attack number for next turn
        else:
            # Invalid phase
            self.done = True
            return self._get_obs(), -10, True, False, {}

        return self._get_obs(), reward, self.done, False, info

    def render(self):
        # Generate a string representation of the current state
        obs = self._get_obs()
        current_player_hp = obs[0] * 50
        opponent_hp = obs[1] * 50
        number_pool = obs[2:12]
        available_numbers = [i + 1 for i in range(10) if number_pool[i] == 1.0]
        phase = "Attack Phase" if obs[12] == 0.0 else "Defense Phase"
        attack_number = obs[13] * 10

        render_str = f"Current Player HP: {current_player_hp}\n"
        render_str += f"Opponent HP: {opponent_hp}\n"
        render_str += f"Available Numbers: {available_numbers}\n"
        render_str += f"Phase: {phase}\n"
        if self.phase == "defense":
            render_str += f"Attack Number to Block: {attack_number}\n"
        return render_str

    def valid_moves(self):
        if self.phase == "attack":
            # Return available numbers in the Number Pool
            return [i + 1 for i in range(10) if self.number_pool[i] == 1.0]
        elif self.phase == "defense":
            # Return possible block amounts between 0 and attack_number
            return list(range(0, self.attack_number + 1))
        else:
            return []

    def _get_obs(self):
        # Generate the observation
        if self.current_player == 0:
            current_player_hp = self.player_hp[0]
            opponent_hp = self.player_hp[1]
        else:
            current_player_hp = self.player_hp[1]
            opponent_hp = self.player_hp[0]

        current_player_hp_scaled = current_player_hp / 50.0
        opponent_hp_scaled = opponent_hp / 50.0
        number_pool = self.number_pool.copy()
        phase_indicator = 0.0 if self.phase == "attack" else 1.0
        attack_number_scaled = self.attack_number / 10.0

        obs = np.concatenate(
            [
                np.array([current_player_hp_scaled, opponent_hp_scaled]),
                number_pool,
                np.array([phase_indicator, attack_number_scaled]),
            ]
        )
        return obs.astype(np.float32)
