import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0: Attack
        # 1: Defend
        # 2: Pass

        self.action_space = spaces.Discrete(3)

        # Observation space: [current_player_energy, current_player_defense_status, opponent_energy, opponent_defense_status]
        # energy from 0 to 10
        # defense_status is 0 or 1

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([10, 1, 10, 1]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize energy units
        self.player_energies = [10, 10]
        # Defense status: 0 means no defend active, 1 means defend active
        self.player_defense = [0, 0]
        # Record last action per player to track defense expiry
        self.last_action_per_player = [None, None]

        # Current player (0 or 1)
        self.current_player = 0

        self.done = False

        # Prepare the initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0
        info = {}

        # Get indices for current player and opponent
        cp = self.current_player
        op = 1 - self.current_player

        # Before processing current player's action, check if their defense effect should expire
        if self.player_defense[cp] == 1 and self.last_action_per_player[op] != 0:
            # Opponent did not attack last turn; defense effect expires
            self.player_defense[cp] = 0

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action, end the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        # Process action
        if action == 0:  # Attack
            # Cost: consumes 2 energy units
            self.player_energies[cp] -= 2

            # Calculate damage
            damage = 3
            if self.player_defense[op]:
                damage -= 2  # Defense reduces damage by 2
                if damage < 0:
                    damage = 0
                # Defense effect expires after being used
                self.player_defense[op] = 0
            # Reduce opponent's energy
            self.player_energies[op] -= damage
            if self.player_energies[op] < 0:
                self.player_energies[op] = 0

            # Check if opponent's energy reached zero
            if self.player_energies[op] == 0:
                self.done = True
                reward = 1  # Current player wins
                return self._get_observation(), reward, True, False, info

        elif action == 1:  # Defend
            # Cost: consumes 1 energy unit
            self.player_energies[cp] -= 1
            # Set defense status
            if self.player_defense[cp] == 0:
                self.player_defense[cp] = 1  # Defense is activated
            # Note: Defense effect does not stack if multiple defenses are performed consecutively

        elif action == 2:  # Pass
            pass  # No cost, no effect

        # After processing action, record the action
        self.last_action_per_player[cp] = action

        # After processing action, check if current player's energy drops below zero
        if self.player_energies[cp] < 0:
            self.player_energies[cp] = 0

        # Switch current player
        self.current_player = op

        # Return observation, reward, done, truncated, info
        observation = self._get_observation()
        return observation, reward, self.done, False, info

    def _get_observation(self):
        cp = self.current_player
        op = 1 - self.current_player
        observation = np.array(
            [
                self.player_energies[cp],
                self.player_defense[cp],
                self.player_energies[op],
                self.player_defense[op],
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        cp = self.current_player
        op = 1 - self.current_player
        s = "--- Energy Duel ---\n"
        s += f"Player {cp+1}'s Turn:\n"
        s += f"Player {cp+1} Energy: {self.player_energies[cp]} "
        s += f"({'Defending' if self.player_defense[cp] else 'Not Defending'})\n"
        s += f"Player {op+1} Energy: {self.player_energies[op]} "
        s += f"({'Defending' if self.player_defense[op] else 'Not Defending'})\n"
        return s

    def valid_moves(self):
        cp = self.current_player
        valid_actions = []
        energy = self.player_energies[cp]
        if energy >= 2:
            valid_actions.append(0)  # Attack
        if energy >= 1:
            valid_actions.append(1)  # Defend
        valid_actions.append(2)  # Pass is always possible
        return valid_actions
