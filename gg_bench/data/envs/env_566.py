import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Attack, 1 - Defend, 2 - Recharge
        self.action_space = spaces.Discrete(3)

        # Define observation space:
        # [current_player_EP, opponent_EP, current_player_defend_status, opponent_defend_status]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.EP = [10, 10]  # Index 0: Player 1 EP, Index 1: Player 2 EP
        self.defend_status = [0, 0]  # 0: no defend, 1: defend active
        self.current_player = 0  # 0: Player 1, 1: Player 2
        self.done = False

        # Return observation and info
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}  # Game already over

        # Check for valid action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid action

        current_player = self.current_player
        opponent = 1 - self.current_player

        # Reset current player's defend status at the start of their turn
        self.defend_status[current_player] = 0

        # Process action
        if action == 0:  # Attack
            # Costs 2 EP
            self.EP[current_player] -= 2
            # Check if current player's EP drops below 0
            if self.EP[current_player] < 0:
                self.EP[current_player] = 0
            # Calculate damage
            damage = 3
            if self.defend_status[opponent] == 1:
                damage -= 2
                damage = max(damage, 1)
            # Apply damage to opponent
            self.EP[opponent] -= damage
            if self.EP[opponent] < 0:
                self.EP[opponent] = 0
        elif action == 1:  # Defend
            # Costs 1 EP
            self.EP[current_player] -= 1
            if self.EP[current_player] < 0:
                self.EP[current_player] = 0
            # Activate defend status
            self.defend_status[current_player] = 1
        elif action == 2:  # Recharge
            # Gain 3 EP, cannot exceed 10
            self.EP[current_player] += 3
            if self.EP[current_player] > 10:
                self.EP[current_player] = 10

        # Check for victory or defeat
        if self.EP[opponent] <= 0 and self.EP[current_player] >= 1:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        elif self.EP[current_player] <= 0:
            # Current player loses
            self.done = True
            return self._get_obs(), -1, True, False, {}

        # No victory or defeat, continue game
        # Switch to next player
        self.current_player = opponent
        return self._get_obs(), 0, False, False, {}

    def render(self):
        player_num = self.current_player + 1
        opponent_num = 2 if player_num == 1 else 1
        render_str = f"Player {player_num}'s Turn:\n"
        render_str += f"Your EP: {self.EP[self.current_player]}\n"
        render_str += f"Opponent's EP: {self.EP[1 - self.current_player]}\n"
        render_str += f"Your Defend Status: {'Active' if self.defend_status[self.current_player] == 1 else 'Inactive'}\n"
        render_str += f"Opponent's Defend Status: {'Active' if self.defend_status[1 - self.current_player] == 1 else 'Inactive'}\n"
        return render_str

    def valid_moves(self):
        valid_actions = []
        current_player = self.current_player
        # Check Attack validity
        if self.EP[current_player] >= 3:
            valid_actions.append(0)
        # Check Defend validity
        if self.EP[current_player] >= 1:
            valid_actions.append(1)
        # Check Recharge validity
        if self.EP[current_player] < 10:
            valid_actions.append(2)
        return valid_actions

    def _get_obs(self):
        current_player = self.current_player
        opponent = 1 - self.current_player
        obs = np.array(
            [
                self.EP[current_player],
                self.EP[opponent],
                self.defend_status[current_player],
                self.defend_status[opponent],
            ],
            dtype=np.float32,
        )
        return obs
