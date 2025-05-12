import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Attack, 1: Defend, 2: Charge
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0]
            ),  # player_HP, opponent_HP, player_charged, opponent_charged
            high=np.array([10, 10, 1, 1]),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_HP = 10
        self.opponent_HP = 10
        self.player_charged = False
        self.opponent_charged = False
        self.done = False
        self.current_player = 0  # 0 for the agent/player, 1 for the opponent
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done or action not in self.valid_moves():
            self.done = True
            return (
                self._get_obs(),
                -10,  # Penalty for invalid action or if game is already done
                True,
                False,
                {},
            )

        # Map action indices to action names
        action_mapping = {0: "Attack", 1: "Defend", 2: "Charge"}
        player_action = action_mapping[action]

        # Simulate opponent's action (can be replaced with another policy for training)
        opponent_action = self._opponent_policy()

        # Resolve actions
        self._resolve_turn(player_action, opponent_action)

        # Check for game end
        if self.player_HP <= 0 and self.opponent_HP > 0:
            self.done = True
            reward = -1  # Player loses
        elif self.opponent_HP <= 0 and self.player_HP > 0:
            self.done = True
            reward = 1  # Player wins
        elif self.player_HP <= 0 and self.opponent_HP <= 0:
            self.done = True
            reward = 0  # Tie (though game shouldn't end in a tie per rules)
        else:
            reward = 0  # Game continues

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        state = f"Player HP: {self.player_HP}, Charged: {'Yes' if self.player_charged else 'No'}\n"
        state += f"Opponent HP: {self.opponent_HP}, Charged: {'Yes' if self.opponent_charged else 'No'}\n"
        return state

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1, 2]  # All actions are always valid unless the game is over

    def _get_obs(self):
        return np.array(
            [
                self.player_HP,
                self.opponent_HP,
                int(self.player_charged),
                int(self.opponent_charged),
            ],
            dtype=np.int32,
        )

    def _opponent_policy(self):
        # Random action for the opponent
        return np.random.choice(["Attack", "Defend", "Charge"])

    def _resolve_turn(self, player_action, opponent_action):
        # Store damage dealt
        player_damage = 0
        opponent_damage = 0

        # Determine damage based on actions
        # Player attacks
        if player_action == "Attack":
            if self.player_charged:
                player_damage = 4  # Charged attack
                self.player_charged = False  # Reset charged state after attack
            else:
                player_damage = 2  # Standard attack

            if opponent_action == "Defend":
                player_damage = 0  # Attack is blocked

            if opponent_action == "Charge":
                opponent_damage += 3  # Opponent is vulnerable while charging

        # Player defends
        elif player_action == "Defend":
            pass  # No damage dealt during defend

        # Player charges
        elif player_action == "Charge":
            self.player_charged = True  # Next attack will be charged
            if opponent_action == "Attack":
                opponent_damage += 3  # Player takes extra damage while charging

        # Opponent attacks
        if opponent_action == "Attack":
            if self.opponent_charged:
                opponent_damage += 4  # Charged attack
                self.opponent_charged = False  # Reset charged state after attack
            else:
                opponent_damage += 2  # Standard attack

            if player_action == "Defend":
                opponent_damage = 0  # Attack is blocked

            if player_action == "Charge":
                player_damage += 3  # Player is vulnerable while charging

        # Opponent defends
        elif opponent_action == "Defend":
            if player_action == "Attack":
                player_damage = 0  # Player's attack is blocked

        # Opponent charges
        elif opponent_action == "Charge":
            self.opponent_charged = True  # Next attack will be charged
            if player_action == "Attack":
                opponent_damage += 3  # Opponent takes extra damage while charging

        # Apply damage
        self.player_HP -= opponent_damage
        self.opponent_HP -= player_damage

        # Ensure HP doesn't go below 0 or above starting HP
        self.player_HP = max(0, min(10, self.player_HP))
        self.opponent_HP = max(0, min(10, self.opponent_HP))
