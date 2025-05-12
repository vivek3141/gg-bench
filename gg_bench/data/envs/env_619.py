import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Three actions: 0 - Attack, 1 - Heal, 2 - Shield
        self.action_space = spaces.Discrete(3)

        # Observation space consists of:
        # [current_player_EP, opponent_EP, current_player_shield, opponent_shield]
        # EP values range from 0 to 10
        # Shield status: 0 (inactive), 1 (active)
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([10, 10, 1, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.EP = [10, 10]  # [current_player_EP, opponent_EP]
        self.shield_active = [False, False]  # [current_player_shield, opponent_shield]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done, False, {}

        # Expire current player's shield at the start of their turn if still active
        if self.shield_active[self.current_player]:
            self.shield_active[self.current_player] = False  # Shield expires

        # Process the action
        if action == 0:  # Attack
            damage = 2
            opponent = 1 - self.current_player
            if self.shield_active[opponent]:
                damage -= 1
                self.shield_active[opponent] = False  # Shield is used up
            damage = max(damage, 0)
            self.EP[opponent] -= damage
            self.EP[opponent] = max(self.EP[opponent], 0)
        elif action == 1:  # Heal
            self.EP[self.current_player] += 1
            self.EP[self.current_player] = min(self.EP[self.current_player], 10)
        elif action == 2:  # Shield
            self.shield_active[self.current_player] = True

        # Check for victory
        opponent = 1 - self.current_player
        if self.EP[opponent] == 0:
            reward = 1
            self.done = True
            return self._get_obs(), reward, self.done, False, {}
        else:
            reward = 0

        # Switch to the next player
        self.current_player = opponent

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        # Observation: [current_player_EP, opponent_EP, current_player_shield, opponent_shield]
        return np.array(
            [
                self.EP[self.current_player],
                self.EP[1 - self.current_player],
                int(self.shield_active[self.current_player]),
                int(self.shield_active[1 - self.current_player]),
            ],
            dtype=np.int32,
        )

    def render(self):
        # Return a string representation of the current game state
        player = self.current_player + 1
        opponent = 1 - self.current_player + 1
        state = f"Player {player}'s Turn:\n"
        state += f"Your EP: {self.EP[self.current_player]}\n"
        state += f"Opponent's EP: {self.EP[1 - self.current_player]}\n"
        if self.shield_active[self.current_player]:
            state += "Your shield is active.\n"
        if self.shield_active[1 - self.current_player]:
            state += "Opponent's shield is active.\n"
        return state

    def valid_moves(self):
        # Return a list of valid actions for the current player
        moves = [0, 2]  # Attack and Shield are always valid
        if self.EP[self.current_player] < 10:
            moves.append(1)  # Heal is valid if EP is less than 10
        return moves
