import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 for Attack, 1 for Defend
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation consists of:
        # [own_HP, own_previous_action, own_consecutive_action_count,
        #  opponent_HP, opponent_previous_action, opponent_consecutive_action_count]
        # own_HP and opponent_HP: between -10 and 10
        # own_previous_action and opponent_previous_action: -1 (None), 0 (Attack), 1 (Defend)
        # own_consecutive_action_count and opponent_consecutive_action_count: between 0 and 2
        low = np.array([-10, -1, 0, -10, -1, 0], dtype=np.int32)
        high = np.array([10, 1, 2, 10, 1, 2], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize game state
        self.player_hp = [10, 10]
        self.previous_action = [-1, -1]  # -1: None, 0: Attack, 1: Defend
        self.consecutive_action_count = [0, 0]  # Count of consecutive same actions
        self.current_player = 0  # Player 0 starts
        self.done = False
        # Damage received from opponent's last Attack
        self.damage_received = [0, 0]
        # Return observation
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        own_idx = self.current_player
        opp_idx = 1 - self.current_player
        observation = np.array(
            [
                self.player_hp[own_idx],
                self.previous_action[own_idx],
                self.consecutive_action_count[own_idx],
                self.player_hp[opp_idx],
                self.previous_action[opp_idx],
                self.consecutive_action_count[opp_idx],
            ],
            dtype=np.int32,
        )
        return observation

    def step(self, action):
        if self.done:
            # Invalid move: game is already over
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        own_idx = self.current_player
        opp_idx = 1 - self.current_player
        reward = 0  # Default reward

        # Update consecutive action count
        if self.previous_action[own_idx] == action:
            self.consecutive_action_count[own_idx] += 1
        else:
            self.consecutive_action_count[own_idx] = 1

        # Update previous action
        self.previous_action[own_idx] = action

        # Apply action
        if action == 0:  # Attack
            # Deal 3 damage to opponent immediately
            damage = 3
            self.player_hp[opp_idx] -= damage
            # Record damage for opponent in case they Defend next turn
            self.damage_received[opp_idx] = damage
        elif action == 1:  # Defend
            # If opponent's last action was Attack, reduce damage by 2
            if self.previous_action[opp_idx] == 0 and self.damage_received[own_idx] > 0:
                damage_reduction = min(2, self.damage_received[own_idx])
                self.player_hp[own_idx] += damage_reduction
                self.damage_received[own_idx] -= damage_reduction
                # HP cannot exceed 10
                if self.player_hp[own_idx] > 10:
                    self.player_hp[own_idx] = 10

        # Check for game end condition
        if self.player_hp[opp_idx] <= 0:
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        if self.player_hp[own_idx] <= 0:
            # Opponent wins
            reward = -1
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Switch current player
        self.current_player = opp_idx

        # Return observation for next player
        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def valid_moves(self):
        own_idx = self.current_player
        last_action = self.previous_action[own_idx]
        count = self.consecutive_action_count[own_idx]
        if count < 2 or last_action == -1:
            return [0, 1]  # Both actions are valid
        else:
            # Must switch action
            if last_action == 0:
                return [1]  # Must Defend
            else:
                return [0]  # Must Attack

    def render(self):
        own_idx = self.current_player
        opp_idx = 1 - self.current_player
        s = f"Player {own_idx + 1}'s turn\n"
        s += f"Player 1 HP: {self.player_hp[0]}\n"
        s += f"Player 2 HP: {self.player_hp[1]}\n"
        actions = {-1: "None", 0: "Attack", 1: "Defend"}
        s += f"Player 1 last action: {actions[self.previous_action[0]]}, consecutive count: {self.consecutive_action_count[0]}\n"
        s += f"Player 2 last action: {actions[self.previous_action[1]]}, consecutive count: {self.consecutive_action_count[1]}\n"
        print(s)
