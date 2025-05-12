import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # There are four possible actions (spells): 0-Fire, 1-Water, 2-Earth, 3-Air
        self.action_space = spaces.Discrete(4)

        # The observation space consists of:
        # 0: phase indicator (0 for attack phase, 1 for defense phase)
        # 1: current player's HP (float between 0 and 10)
        # 2: opponent's HP (float between 0 and 10)
        # 3-6: current player's available spells (1 if available, 0 if used)
        # 7-10: opponent's used spells (1 if used, 0 if not used)
        # 11-14: attacker's spell (one-hot encoding, zeros if not applicable)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(15,), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [10, 10]  # Player 0 and Player 1 HP
        self.player_available_spells = [[1, 1, 1, 1], [1, 1, 1, 1]]  # For both players
        self.player_used_spells = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.attacker_spell = None  # No spell selected yet
        # Randomly select starting player (0 or 1)
        self.current_player = random.randint(0, 1)
        self.phase = "attack"
        self.done = False

        # Build the initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        phase_indicator = 0 if self.phase == "attack" else 1
        own_hp = self.player_hp[self.current_player]
        opp_hp = self.player_hp[1 - self.current_player]
        own_available_spells = self.player_available_spells[self.current_player]
        opp_used_spells = self.player_used_spells[1 - self.current_player]
        # Attacker's spell (one-hot encoding)
        if self.attacker_spell is not None:
            attacker_spell_one_hot = [0, 0, 0, 0]
            attacker_spell_one_hot[self.attacker_spell] = 1
        else:
            attacker_spell_one_hot = [0, 0, 0, 0]
        observation = np.array(
            [phase_indicator, own_hp, opp_hp]
            + own_available_spells
            + opp_used_spells
            + attacker_spell_one_hot,
            dtype=np.float32,
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if player has any valid moves
        if len(self.valid_moves()) == 0:
            # Player cannot perform an action, loses the game
            reward = 0
            self.done = True
            return self._get_observation(), reward, True, False, {}

        if action not in self.valid_moves():
            # Invalid move
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        reward = 0
        current_player = self.current_player
        opponent = 1 - self.current_player

        if self.phase == "attack":
            # Attacker selects a spell
            self.attacker_spell = action
            # Mark spell as used
            self.player_available_spells[current_player][action] = 0
            self.player_used_spells[current_player][action] = 1
            # Switch phase to defense
            self.phase = "defense"
            return self._get_observation(), reward, False, False, {}
        elif self.phase == "defense":
            # Defender selects a spell
            defender_spell = action
            # Mark spell as used
            self.player_available_spells[current_player][action] = 0
            self.player_used_spells[current_player][action] = 1

            # Resolve the outcome
            outcome = self._resolve_clash(self.attacker_spell, defender_spell)
            if outcome == 1:
                # Attacker wins
                self.player_hp[opponent] -= 2  # Defender loses 2 HP
            elif outcome == -1:
                # Defender wins
                self.player_hp[current_player] -= 1  # Attacker loses 1 HP
            elif outcome == 0:
                # Tie
                self.player_hp[current_player] -= 1  # Both lose 1 HP
                self.player_hp[opponent] -= 1

            # HP cannot be negative
            self.player_hp[current_player] = max(0, self.player_hp[current_player])
            self.player_hp[opponent] = max(0, self.player_hp[opponent])

            # Check for victory conditions
            if self.player_hp[opponent] == 0 and self.player_hp[current_player] == 0:
                # Both HP are 0, attacker wins
                reward = 1  # Current player wins
                self.done = True
            elif self.player_hp[opponent] == 0:
                # Opponent loses
                reward = 1  # Current player wins
                self.done = True
            elif self.player_hp[current_player] == 0:
                # Current player loses
                reward = 0
                self.done = True
            else:
                # Game continues
                reward = 0
                # Switch phase to attack and switch current player
                self.phase = "attack"
                self.current_player = opponent
                self.attacker_spell = None

                # Check if new current player has any valid moves
                if len(self.valid_moves()) == 0:
                    # Player cannot perform an action, loses the game
                    self.done = True

            return self._get_observation(), reward, self.done, False, {}

    def _resolve_clash(self, attacker_spell, defender_spell):
        # Define the interaction matrix
        # 0: Tie, 1: Attacker wins, -1: Defender wins
        interaction_matrix = np.array(
            [
                # Defender's spells: Fire Water Earth Air
                [0, -1, 1, 1],  # Attacker's spell is Fire
                [1, 0, 1, -1],  # Attacker's spell is Water
                [-1, -1, 0, 1],  # Attacker's spell is Earth
                [-1, 1, -1, 0],  # Attacker's spell is Air
            ]
        )
        return interaction_matrix[attacker_spell, defender_spell]

    def valid_moves(self):
        # Return the indices of available spells for the current player
        available_spells = self.player_available_spells[self.current_player]
        valid_moves = [i for i in range(4) if available_spells[i] == 1]
        return valid_moves

    def render(self):
        current_player = self.current_player
        opponent = 1 - current_player
        phase = self.phase
        s = f"Current player: Player {current_player + 1}\n"
        s += f"Phase: {phase}\n"
        s += f"Player {current_player + 1} HP: {self.player_hp[current_player]}\n"
        s += f"Player {opponent + 1} HP: {self.player_hp[opponent]}\n"

        s += f"Player {current_player + 1} available spells: "
        spells = ["Fire", "Water", "Earth", "Air"]
        available_spells = [
            spells[i]
            for i in range(4)
            if self.player_available_spells[current_player][i] == 1
        ]
        s += ", ".join(available_spells) + "\n"

        s += f"Player {opponent + 1} used spells: "
        used_spells = [
            spells[i] for i in range(4) if self.player_used_spells[opponent][i] == 1
        ]
        s += ", ".join(used_spells) + "\n"

        if self.phase == "defense" and self.attacker_spell is not None:
            s += f"Attacker's spell: {spells[self.attacker_spell]}\n"

        return s
