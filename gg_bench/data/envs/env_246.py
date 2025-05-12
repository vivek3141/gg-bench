import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Attack values from 1 to 5, represented as actions 0 to 4
        self.action_space = spaces.Discrete(5)

        # Observation space consists of:
        # [Player HP, Opponent HP, Player usages of attacks 1-5, Opponent usages of attacks 1-5]
        self.observation_space = spaces.Box(
            low=np.array([0] * 12), high=np.array([10, 10] + [2] * 10), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = 10
        self.opponent_hp = 10
        # Attack usages for attack values 1 to 5 (indices 0 to 4)
        self.player_usages = np.array([2, 2, 2, 2, 2], dtype=np.int32)
        self.opponent_usages = np.array([2, 2, 2, 2, 2], dtype=np.int32)
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate(
            (
                np.array([self.player_hp], dtype=np.int32),
                np.array([self.opponent_hp], dtype=np.int32),
                self.player_usages,
                self.opponent_usages,
            )
        )

    def step(self, action):
        if self.done:
            # The game is over
            return self._get_obs(), 0, self.done, False, {}

        # Check if the player has any valid moves
        if len(self.valid_moves()) == 0:
            # Player has no valid moves, loses
            self.done = True
            reward = -10
            return self._get_obs(), reward, self.done, False, {}

        # Map action index to attack value
        attack_value = action + 1  # action 0 corresponds to attack value 1

        # Check if the player's action is valid
        if self.player_usages[action] <= 0:
            # Invalid move, player loses automatically
            self.done = True
            reward = -10
            return self._get_obs(), reward, self.done, False, {}

        # Decrement player's usage of selected attack
        self.player_usages[action] -= 1

        # Opponent selects action
        valid_opponent_actions = [i for i in range(5) if self.opponent_usages[i] > 0]

        # Check if opponent has valid moves
        if len(valid_opponent_actions) == 0:
            # Opponent has no valid moves, player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, self.done, False, {}

        opponent_action = np.random.choice(valid_opponent_actions)
        opponent_attack_value = opponent_action + 1

        # Decrement opponent's usage of selected attack
        self.opponent_usages[opponent_action] -= 1

        # Resolve damage
        if attack_value == opponent_attack_value:
            # No damage dealt by either player
            pass
        elif attack_value > opponent_attack_value:
            # Player deals damage to opponent
            damage = attack_value - opponent_attack_value
            self.opponent_hp -= damage
            if self.opponent_hp <= 0:
                self.opponent_hp = 0
                self.done = True
                reward = 1
                return self._get_obs(), reward, self.done, False, {}
        else:
            # Opponent deals damage to player
            damage = opponent_attack_value - attack_value
            self.player_hp -= damage
            if self.player_hp <= 0:
                self.player_hp = 0
                self.done = True
                reward = 0
                return self._get_obs(), reward, self.done, False, {}

        # Game continues
        reward = 0
        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        render_str = f"Player HP: {self.player_hp}\n"
        render_str += f"Opponent HP: {self.opponent_hp}\n"
        render_str += "Player remaining usages of attacks 1-5:\n"
        render_str += (
            " ".join(f"{i+1}({self.player_usages[i]})" for i in range(5)) + "\n"
        )
        render_str += "Opponent remaining usages of attacks 1-5:\n"
        render_str += " ".join(f"{i+1}({self.opponent_usages[i]})" for i in range(5))
        return render_str

    def valid_moves(self):
        return [i for i in range(5) if self.player_usages[i] > 0]
