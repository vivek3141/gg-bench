import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 4 possible actions (spells)
        self.action_space = spaces.Discrete(4)

        # Observation space:
        # Each player has:
        # - MP (0-10)
        # - Remaining spell uses: Fireball(0-3), Shield(0-2), Drain(0-2), Recharge(0-2)
        # - shield_active: 0 or 1
        # - shield_used: 0 or 1
        # So total of 7 elements per player
        low = np.array([0] * 14, dtype=np.int32)
        high = np.array([10, 3, 2, 2, 2, 1, 1] * 2, dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the game state
        self.state = np.array(
            [10, 3, 2, 2, 2, 0, 0, 10, 3, 2, 2, 2, 0, 0],  # Player 0  # Player 1
            dtype=np.int32,
        )
        self.current_player = 0  # 0 or 1
        self.done = False
        return self.state.copy(), {}

    def step(self, action):
        if self.done:
            return self.state.copy(), 0, True, False, {}

        # Handle shield expiration for current player
        player_offset = self.current_player * 7
        opponent_offset = (1 - self.current_player) * 7

        # Expire shield if needed
        if self.state[player_offset + 5] == 1 and self.state[player_offset + 6] == 0:
            # Shield wasn't used, expires
            self.state[player_offset + 5] = 0  # shield_active
            self.state[player_offset + 6] = 0  # shield_used
        elif self.state[player_offset + 5] == 1 and self.state[player_offset + 6] == 1:
            # Shield was used, reset
            self.state[player_offset + 5] = 0  # shield_active
            self.state[player_offset + 6] = 0  # shield_used

        # Check if action is valid
        spell_uses = self.state[
            player_offset + 1 : player_offset + 5
        ]  # Spells: Fireball, Shield, Drain, Recharge

        if action < 0 or action > 3:
            # Invalid action index
            self.done = True
            return (
                self.state.copy(),
                -10,
                True,
                False,
                {},
            )  # Reward -10 for invalid action

        if spell_uses[action] <= 0:
            # Invalid action: no uses left
            self.done = True
            return (
                self.state.copy(),
                -10,
                True,
                False,
                {},
            )  # Reward -10 for invalid action

        # Action is valid, apply spell effect
        # Subtract one use from the spell
        self.state[player_offset + 1 + action] -= 1

        reward = 0
        info = {}
        terminated = False
        truncated = False

        # Spell effects
        if action == 0:  # Fireball
            # Deals 3 damage to the opponent
            if self.state[opponent_offset + 5] == 1:
                # Opponent has an active shield, damage is negated, shield is used
                self.state[opponent_offset + 6] = 1  # shield_used
            else:
                # Subtract 3 from opponent's MP
                self.state[opponent_offset] -= 3
                # Check if opponent's MP reaches zero or below
                if self.state[opponent_offset] <= 0:
                    self.done = True
                    reward = 1
                    return self.state.copy(), reward, True, False, {}
        elif action == 1:  # Shield
            # Grants a shield
            self.state[player_offset + 5] = 1  # shield_active
            self.state[player_offset + 6] = 0  # shield_used
        elif action == 2:  # Drain
            # Steals 2 MP from opponent
            if self.state[opponent_offset + 5] == 1:
                # Opponent has an active shield, drain is negated, shield is used
                self.state[opponent_offset + 6] = 1  # shield_used
            else:
                steal_amount = min(2, self.state[opponent_offset])
                self.state[opponent_offset] -= steal_amount
                self.state[player_offset] += steal_amount
                # MP cannot exceed 10
                if self.state[player_offset] > 10:
                    self.state[player_offset] = 10
                # Check if opponent's MP reaches zero or below
                if self.state[opponent_offset] <= 0:
                    self.done = True
                    reward = 1
                    return self.state.copy(), reward, True, False, {}
        elif action == 3:  # Recharge
            # Restores 2 MP to player
            self.state[player_offset] += 2
            if self.state[player_offset] > 10:
                self.state[player_offset] = 10

        # Switch to next player
        self.current_player = 1 - self.current_player
        done = self.done
        return self.state.copy(), reward, done, False, {}

    def render(self):
        state_str = ""
        for i in range(2):
            player_offset = i * 7
            mp = self.state[player_offset]
            spell_uses = self.state[player_offset + 1 : player_offset + 5]
            shield_active = self.state[player_offset + 5]
            shield_used = self.state[player_offset + 6]
            spells = ["Fireball", "Shield", "Drain", "Recharge"]
            state_str += f"Player {i}:\n"
            state_str += f"  MP: {mp}\n"
            state_str += "  Spells:\n"
            for j, spell in enumerate(spells):
                state_str += f"    {spell}: {spell_uses[j]} uses left\n"
            state_str += f"  Shield active: {'Yes' if shield_active else 'No'}\n"
        return state_str

    def valid_moves(self):
        player_offset = self.current_player * 7
        spell_uses = self.state[player_offset + 1 : player_offset + 5]
        valid_moves = [i for i in range(4) if spell_uses[i] > 0]
        return valid_moves
