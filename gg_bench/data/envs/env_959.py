import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (defend), 1-10 (attack with values 1-10)
        self.action_space = spaces.Discrete(11)  # Actions 0 to 10

        # Define observation space
        # Observation consists of:
        # 0: own HP (normalized 0-1)
        # 1: opponent's HP (normalized 0-1)
        # 2-11: own available attacks (1 if available, 0 if used)
        # 12-21: opponent's used attacks (1 if used, 0 if not used)
        # 22: own defend cooldown status (1 if cannot defend, 0 if can defend)
        # 23: opponent's defend status (1 if opponent is defending, 0 otherwise)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(24,), dtype=np.float32
        )

        self.current_player = 1
        self.other_player = 2

        self.player_hp = {1: 100, 2: 100}
        self.available_attacks = {
            1: np.ones(10, dtype=int),
            2: np.ones(10, dtype=int),
        }
        self.defend_status = {1: False, 2: False}
        self.defend_cooldown = {1: False, 2: False}
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1
        self.other_player = 2

        self.player_hp = {1: 100, 2: 100}
        self.available_attacks = {
            1: np.ones(10, dtype=int),
            2: np.ones(10, dtype=int),
        }
        self.defend_status = {1: False, 2: False}
        self.defend_cooldown = {1: False, 2: False}
        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.done:
            return self._get_observation(), 0, True, False, info

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Process action
        if action == 0:
            # Defend action
            self.defend_status[self.current_player] = True
            self.defend_cooldown[self.current_player] = True
        else:
            # Attack action
            attack_value = action  # Action corresponds to attack value 1-10
            index = attack_value - 1  # Indices 0-9

            if self.available_attacks[self.current_player][index] == 0:
                # Attack value already used
                reward = -10
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, info

            # Use the attack value
            self.available_attacks[self.current_player][index] = 0

            # Calculate damage
            if self.defend_status[self.other_player]:
                damage = attack_value // 2
                self.defend_status[self.other_player] = False  # Defend effect used
            else:
                damage = attack_value

            # Apply damage to opponent
            self.player_hp[self.other_player] -= damage
            if self.player_hp[self.other_player] < 0:
                self.player_hp[self.other_player] = 0

        # Update defend cooldown
        if self.defend_cooldown[self.current_player]:
            self.defend_cooldown[self.current_player] = False

        # Check for victory
        if self.player_hp[self.other_player] <= 0:
            reward = 1
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Switch players
        self.current_player, self.other_player = self.other_player, self.current_player

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        output = "----------------------------------------\n"
        output += f"It's Player {self.current_player}'s turn!\n"
        output += f"Your HP: {self.player_hp[self.current_player]}\n"
        output += f"Opponent's HP: {self.player_hp[self.other_player]}\n"

        available_attacks = [
            str(i + 1)
            for i in range(10)
            if self.available_attacks[self.current_player][i] == 1
        ]
        output += f"Your Available Attacks: {', '.join(available_attacks)}\n"

        actions = []
        if not self.defend_cooldown[self.current_player]:
            actions.append("defend")
        if available_attacks:
            actions.append("attack")
        output += f"Actions Available: {', '.join(actions)}\n"

        if self.defend_cooldown[self.current_player]:
            output += "(You cannot defend this turn.)\n"

        if self.defend_status[self.other_player]:
            output += "(Opponent is defending. Their next incoming attack damage will be halved.)\n"

        output += "----------------------------------------\n"
        return output

    def valid_moves(self):
        valid_actions = []

        if not self.defend_cooldown[self.current_player]:
            valid_actions.append(0)  # Defend action

        for idx in range(10):
            if self.available_attacks[self.current_player][idx] == 1:
                valid_actions.append(idx + 1)  # Attack actions 1-10

        return valid_actions

    def _get_observation(self):
        obs = np.zeros(24, dtype=np.float32)

        obs[0] = self.player_hp[self.current_player] / 100.0
        obs[1] = self.player_hp[self.other_player] / 100.0

        obs[2:12] = self.available_attacks[self.current_player]

        obs[12:22] = 1 - self.available_attacks[self.other_player]

        obs[22] = float(self.defend_cooldown[self.current_player])

        obs[23] = float(self.defend_status[self.other_player])

        return obs
