import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Charge Laser, 1 = Fire Laser
        self.action_space = spaces.Discrete(2)
        # Observation: [my_shield_strength, my_laser_charge, opponent_shield_strength, opponent_laser_charge]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([10, 5, 10, 5]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shield_strengths = np.array(
            [10, 10], dtype=np.int32
        )  # [Player 1, Player 2]
        self.laser_charges = np.array([0, 0], dtype=np.int32)  # [Player 1, Player 2]
        self.current_player = 0  # Player 1 starts (index 0)
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Process the action
        if action == 0:  # Charge Laser
            self.laser_charges[self.current_player] += 1
        elif action == 1:  # Fire Laser
            damage = self.laser_charges[self.current_player]
            self.laser_charges[self.current_player] = 0
            opponent = 1 - self.current_player
            self.shield_strengths[opponent] -= damage

            # Check for win condition
            if self.shield_strengths[opponent] <= 0:
                self.done = True
                return self._get_obs(), 1, True, False, {}

        # Reward for a valid move is -10
        reward = -10

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return self._get_obs(), reward, False, False, {}

    def render(self):
        opponent = 1 - self.current_player
        render_str = f"Player {self.current_player + 1}'s turn:\n"
        render_str += (
            f"Your Shield Strength: {self.shield_strengths[self.current_player]}\n"
        )
        render_str += f"Your Laser Charge: {self.laser_charges[self.current_player]}\n"
        render_str += f"Opponent's Shield Strength: {self.shield_strengths[opponent]}\n"
        render_str += f"Opponent's Laser Charge: {self.laser_charges[opponent]}\n"
        return render_str

    def valid_moves(self):
        moves = []
        laser_charge = self.laser_charges[self.current_player]
        if laser_charge == 0:
            moves.append(0)  # Only Charge Laser is valid
        elif laser_charge == 5:
            moves.append(1)  # Must Fire Laser
        else:
            moves.extend([0, 1])  # Can Charge or Fire
        return moves

    def _get_obs(self):
        opponent = 1 - self.current_player
        obs = np.array(
            [
                self.shield_strengths[self.current_player],
                self.laser_charges[self.current_player],
                self.shield_strengths[opponent],
                self.laser_charges[opponent],
            ],
            dtype=np.int32,
        )
        return obs
