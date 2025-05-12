import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0-4 for attack values 1-5, 5 for Block, 6 for Counter-Attack
        self.action_space = spaces.Discrete(7)

        # Observation space contains:
        #   Current player's HP (0-10)
        #   Opponent's HP (0-10)
        #   Phase (0 for Attack Phase, 1 for Defense Phase)
        #   Attack value (0 if not applicable, else 1-5)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0, 0]),
            high=np.array([10.0, 10.0, 1, 5]),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [10.0, 10.0]  # Player 0 and Player 1 HP
        self.current_player = 0  # Player 0 starts
        self.phase = 0  # 0 for Attack Phase, 1 for Defense Phase
        self.attack_value = 0
        self.done = False
        # Observation: [current_player_hp, opponent_hp, phase, attack_value]
        observation = np.array(
            [
                self.player_hp[self.current_player],
                self.player_hp[1 - self.current_player],
                self.phase,
                self.attack_value,
            ],
            dtype=np.float32,
        )
        return observation, {}

    def step(self, action):
        if self.done:
            # If the game is over, no more actions can be taken
            return self._get_obs(), -10, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {"error": "Invalid action"}

        reward = -10  # Default reward for a valid move

        if self.phase == 0:
            # Attack Phase
            # Action must be between 0 and 4
            self.attack_value = action + 1  # Map action to attack value (1-5)

            # Switch to Defense Phase
            self.phase = 1
            # Do not switch player; defender is the opponent
            # Observation will be updated accordingly
        elif self.phase == 1:
            # Defense Phase
            # Action must be 5 (Block) or 6 (Counter-Attack)
            defender = self.current_player
            attacker = 1 - self.current_player

            if action == 5:
                # Block
                damage = max(self.attack_value - 2, 0)
                self.player_hp[defender] -= damage
            elif action == 6:
                # Counter-Attack
                self.player_hp[defender] -= 1  # Lose 1 HP for counter-attacking
                self.player_hp[attacker] -= self.attack_value  # Reflect full attack

            # Check for defeat
            if self.player_hp[defender] <= 0 and self.player_hp[attacker] <= 0:
                # Both players have HP <= 0; current player loses
                self.done = True
                reward = -10
            elif self.player_hp[defender] <= 0:
                # Defender (current player) loses
                self.done = True
                reward = -10
            elif self.player_hp[attacker] <= 0:
                # Attacker (opponent) loses; current player wins
                self.done = True
                reward = 1

            # Reset attack value
            self.attack_value = 0
            # Switch phase back to Attack Phase
            self.phase = 0
            # Switch current player to attacker (next turn)
            self.current_player = attacker

        observation = self._get_obs()
        terminated = self.done
        truncated = False  # No truncation in this game
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        player0_role = (
            "Attacker"
            if self.current_player == 0 and self.phase == 0
            else "Defender" if self.current_player == 0 else ""
        )
        player1_role = (
            "Attacker"
            if self.current_player == 1 and self.phase == 0
            else "Defender" if self.current_player == 1 else ""
        )
        output = f"Player 0 HP: {self.player_hp[0]:.1f} {player0_role}\n"
        output += f"Player 1 HP: {self.player_hp[1]:.1f} {player1_role}\n"
        if self.phase == 0:
            output += "Phase: Attack Phase\n"
            output += f"Current Player: Player {self.current_player} (Attacker)\n"
            output += "Choose an attack value between 1 and 5.\n"
        elif self.phase == 1:
            output += "Phase: Defense Phase\n"
            output += f"Current Player: Player {self.current_player} (Defender)\n"
            output += f"Attack Value: {self.attack_value}\n"
            output += "Choose to Block or Counter-Attack.\n"
        return output

    def valid_moves(self):
        if self.phase == 0:
            # Attack Phase: valid actions are 0-4 (attack values 1-5)
            return [0, 1, 2, 3, 4]
        elif self.phase == 1:
            # Defense Phase: valid actions are 5 (Block) and 6 (Counter-Attack)
            return [5, 6]
        else:
            return []

    def _get_obs(self):
        # Observation: [current_player_hp, opponent_hp, phase, attack_value]
        observation = np.array(
            [
                self.player_hp[self.current_player],
                self.player_hp[1 - self.current_player],
                self.phase,
                self.attack_value,
            ],
            dtype=np.float32,
        )
        return observation
